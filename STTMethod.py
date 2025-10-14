import numpy as np
from scipy.integrate import solve_ivp
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import matplotlib.pyplot as plt

class PropagateSatellite():
    # Constants
    mu_e    = 398600.44    # km**3/2**2
    R_e     = 6378.137      # km
    J2      = 1.08264e-3

    PARAM_DIM = 10
    STATE_DIM = 6
    MEASUREMENT_DIM = 2
    nu = 1e-3
    i_max = 10

    P_0 = np.diag([10**2, 10**2, 10**2, 1e-3**2, 1e-3**2, 1e-3**2, 5e-3**2, 5e-3**2, 5e-3**2, 50**2])
    sigma_noise = 1e-4
    R = sigma_noise**2 * np.eye(MEASUREMENT_DIM)

    results = []

# Dirac delta function since its not differentiable
# Use Gaussian function over that time period
# Turn generated points into elevation and azimuth

    def __init__(self, num_measurement, time_split, x_dv, y_dv, z_dv, t_dv, xi_dv, yi_dv, zi_dv, ti_dv, record = False):
        self.record = record

        self.K = num_measurement
        self.t_eval = np.arange(0, (num_measurement + 1)* time_split, time_split)

        self.observer = self.generate_input_points(500, 0.01, 45.05, 29.93, 132.9, -107.74, 0, 0, 0, 0, 0, self.t_eval)
        self.target = self.generate_input_points(1000, 0.02, 45, 94.80, 199.00, -54.13, 1e-5, x_dv, y_dv, z_dv, t_dv, self.t_eval)
        self.X_0 = np.hstack((self.target[0] + [10, 10, 10, 5e-3, 5e-3, 5e-3], np.array([xi_dv/1000, yi_dv/1000, zi_dv/1000, ti_dv])))

        print("I have generated the observer and target points")

        self.X_i = self.X_0
        self.P_i = self.P_0

        self.z_tau = np.zeros((self.K, 2))
        self.z_exp = np.zeros((self.K, 2))

        for i in range(self.K):
            self.z_tau[i] = self.transform_state(self.target[i], self.observer[i])

        print("I have generated the expected points")

        # Covariance matrix
        self.R_2 = self.sigma_noise**2 * np.eye(self.K * self.MEASUREMENT_DIM)

    def transform_state(self, x_target, x_observer):
        r_obs = x_observer[:3]
        v_obs = x_observer[3:6]
        rel = x_target[:3] - r_obs

        R_hat = r_obs / np.linalg.norm(r_obs)
        H = np.cross(r_obs, v_obs)
        C_hat = H / np.linalg.norm(H)
        T_hat = np.cross(C_hat, R_hat)

        R_orbital = np.vstack((T_hat, C_hat, R_hat))
        rel_local = R_orbital @ rel

        x, y, z = rel_local
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / np.linalg.norm(rel_local))

        return np.array([azimuth, elevation])

    def dynamics(self, t, state):
        r = state[:3]
        v = state[3:]

        # J2 Perturbation
        a_j2r = [self.mu_e * r[0] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-3/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2),
                self.mu_e * r[1] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-3/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2),
                self.mu_e * r[2] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-9/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2)
                ]
        
        norm_r = np.linalg.norm(r)
        a = -self.mu_e * r/norm_r**3 + a_j2r  + np.exp(-((t -  self.X_i[9])**2) / (2 * 10**2)) * self.X_i[6:9] / 10
        return np.concatenate([v, a])

    def generate_input_points(self, a, e, i, raan, argp, nu, noise, dvx, dvy, dvz, man_time, t_eval):
        # Sample orbit test
        epoch = Time("2025-01-01 00:00:00", scale="utc")
        a = (a + self.R_e) * u.km
        e *= u.one
        i *= u.deg
        raan *= u.deg
        argp *= u.deg
        nu *= u.deg
        dv = [dvx/1000, dvy/1000, dvz/1000] * u.km / u.s

        orbit = Orbit.from_classical(Earth, a, e, i, raan, argp, nu, epoch)

        positions = []
        velocities = []
        man_applied = False

        for t in t_eval:
            orbit = orbit.propagate(10 * u.s)
            if t >= man_time and not man_applied:
                # Apply impulse
                new_velocity = orbit.v + dv
                orbit = Orbit.from_vectors(Earth, orbit.r, new_velocity, epoch + t * u.s)
                man_applied = True

            r = orbit.r.to_value(u.km)
            v = orbit.v.to_value(u.km / u.s)

            # Add Gaussian noise
            r_noisy = r + np.random.normal(0, noise, size=3)
            v_noisy = v + np.random.normal(0, noise, size=3)

            positions.append(r_noisy)
            velocities.append(v_noisy)

        positions = np.array(positions)
        velocities = np.array(velocities)
        states = np.hstack((positions, velocities))
        return states[:-1]

    #dy/dx
    def calc_first_diff(self, point, t):
        epsilon = 1e-6
        y_state = point[:6]
        dy_dx = np.zeros((self.STATE_DIM, self.STATE_DIM))

        for i in range(self.STATE_DIM):
            dx = np.zeros(self.STATE_DIM)
            dx[i] = epsilon
            y_plus = self.dynamics(t, y_state + dx)
            y_minus = self.dynamics(t, y_state - dx)
            dy_dx[:, i] = (y_plus - y_minus) / (2 * epsilon)

        return dy_dx

    #d^2y/dx^2
    def calc_second_diff(self, point, t):
        epsilon = 1e-6
        y_state = point[:6]
        d2y_dx2 = np.zeros((self.STATE_DIM, self.STATE_DIM, self.STATE_DIM))

        for i in range(self.STATE_DIM):
            for j in range(self.STATE_DIM):
                dx_i = np.zeros(self.STATE_DIM)
                dx_j = np.zeros(self.STATE_DIM)
                dx_i[i] = epsilon
                dx_j[j] = epsilon

                y_pp = self.dynamics(t, y_state + dx_i + dx_j)
                y_pm = self.dynamics(t, y_state + dx_i - dx_j)
                y_mp = self.dynamics(t, y_state - dx_i + dx_j)
                y_mm = self.dynamics(t, y_state - dx_i - dx_j)

                d2y_dx2[:, i, j] = (y_pp - y_pm - y_mp + y_mm) / (4 * epsilon**2)

        return d2y_dx2


    def phi_dot(self, t, phi_flat):
        phi = phi_flat.reshape(self.STATE_DIM, self.STATE_DIM)
        dphi = self.y_ref_phi[int(t / 10), :] @ phi
        return dphi.flatten()
    
    def psi_dot(self, t, psi_flat):
        psi = psi_flat.reshape(self.STATE_DIM, self.STATE_DIM, self.STATE_DIM)
        dpsi = self.y_ref_psi[int(t / 10), :, :] @ psi + self.y_ref_phi[int(t / 10), :] @ self.A_1[int(t / 10), :] @ self.A_1[int(t / 10), :]
        return dpsi.flatten()
    
    def compute_stt(self, t_eval, X, propagate_with_impulse, epsilon_vector = None):
        if epsilon_vector is None:
            epsilon_vector = np.array([1, 1, 1, 1, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3, 1e-2])  # r/v, Δv, t_impulse

        phi = np.zeros((self.STATE_DIM, self.PARAM_DIM))
        psi = np.zeros((self.STATE_DIM, self.PARAM_DIM, self.PARAM_DIM))
        #phi_tmp = np.zeros((self.STATE_DIM, self.STATE_DIM))
        psi_tmp = np.zeros((self.STATE_DIM, self.STATE_DIM, self.STATE_DIM))
        self.A_1 = np.zeros((self.STATE_DIM, self.STATE_DIM))
        A_2 = np.zeros((self.STATE_DIM, self.STATE_DIM, self.STATE_DIM))
        I = np.identity(self.STATE_DIM)
        
        t_pre = t_eval[t_eval <= X[9]]
        t_post = t_eval[t_eval >= X[9]]

        # Phi from t_0 - t (ie A_1)
        phi0 = np.eye(self.STATE_DIM).flatten()
        self.y_ref_phi = np.array([self.calc_first_diff(X, t) for t in t_eval])
        sol_phi = solve_ivp(self.phi_dot, [0, t_pre[-1]], phi0, t_eval=t_pre, method='DOP853', rtol=1e-10, atol=1e-12)
        #self.phi_t0 = np.pad(sol_phi.y.T.reshape(len(t_pre), 6, 6), ((0, 0), (0, 0), (0, 4)), mode='constant')
        #self.phi_tmp = sol_phi.y.T.reshape(t_eval, 6, 6)
        self.A_1 = sol_phi.y.T.reshape(t_eval, 6, 6)

        # Psi from t_0 - t (ie A_2)
        psi0 = np.zeros((self.STATE_DIM, self.STATE_DIM, self.STATE_DIM)).flatten()
        self.y_ref_psi = np.array([self.calc_second_diff(X, t) for t in t_eval])
        sol_psi = solve_ivp(self.psi_dot, [0, t_eval], psi0, t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
        #A_2 = np.pad(sol_psi.y.T.reshape(len(t_pre), 6, 6, 6), ((0, 0), (0, 0), (0, 4), (0, 4)), mode='constant')
        A_2 = sol_psi.y.T.reshape(t_eval, 6, 6, 6)

        # t - t_2
        #x_plus is state after manoeuvre
        #x_minus is state before manoeuvre
        X_no_dv = np.hstack((X[:6], 0, 0, 0, 1800))
        x_minus = np.hstack((propagate_with_impulse(t_pre, X)[-1], X[6:]))
        x_plus= np.hstack((propagate_with_impulse(t_pre, X_no_dv)[-1], X[6:]))

        # # f
        f_plus = propagate_with_impulse(t_post, x_minus)
        f_minus = propagate_with_impulse(t_post, x_plus)

        # # g
        # g_minus_p = propagate_with_impulse(np.hstack(([t_post[0] + epsilon_vector[9]], t_post[1:])), X_no_man)
        # g_minus_m = propagate_with_impulse(np.hstack(([t_post[0] - epsilon_vector[9]], t_post[1:])), X_no_man)
        # g_minus = (g_minus_p - g_minus_m) / (2 * epsilon_vector[9])

        # g_plus_p = propagate_with_impulse(np.hstack(([t_post[0] + epsilon_vector[9]], t_post[1:])), X_t1)
        # g_plus_m = propagate_with_impulse(np.hstack(([t_post[0] - epsilon_vector[9]], t_post[1:])), X_t1)
        # g_plus = (g_plus_p - g_plus_m) / (2 * epsilon_vector[9])

        # # H
        H_plus = np.array([self.calc_first_diff(x_plus, t) for t in t_post])
        H_minus = np.array([self.calc_first_diff(x_minus, t) for t in t_post])

        B_1 =  f_minus - f_plus
        # B_2 = g_minus - g_plus + 2 * np.einsum('kp,p ->k', H_plus, f_plus)
        C = H_minus - np.einsum('ikl,ilp->ikp', H_plus, self.A_1)
        D = -H_plus


        # Compile matrices
        # Phi
        for i in range(self.STATE_DIM):
            phi[:, i] = self.phi_tmp @ self.A_1[:, i]
        phi[:, 6:9] = self.phi_tmp[:, 3:6]
        phi[:, 9] = self.phi_tmp @ B_1

        # Psi
        psi[:, 6:9, 6:9] = psi_tmp[:, 3:6, 3:6]

        # 0 - 6 x 0 - 6
        for i in range(self.STATE_DIM):
            for j in range(self.STATE_DIM):
                psi[:, i, j] = np.einsum('ijk, j, k->i', psi_tmp, self.A_1[:, i], self.A_1[:, j]) + self.phi_tmp @ A_2[:, i, j]

            for j in range(6, 9):
                psi[:, i, j - 3] = np.einsum('ijk, j, k->i', psi_tmp, self.A_1[:, i], I[:, j - 3])
                psi[:, j, i - 3] = np.einsum('ijk, j, k->i', psi_tmp, self.A_1[:, i], I[:, j - 3])

            psi[:, i, 9] = np.einsum('ijk, j, k->i', psi_tmp, self.A_1[:, i], B_1) + self.phi_tmp @ C[:, i]
            psi[:, 9, i] = np.einsum('ijk, j, k->i', psi_tmp, self.A_1[:, i], B_1) + self.phi_tmp @ C[:, i]
        
        for i in range(6, 9):
            psi[:, i, 9] = psi_tmp[:, :, i - 3] @ B_1 + self.phi_tmp @ D[:, i - 3]
            psi[:, 9, i] = psi_tmp[:, :, i - 3] @ B_1 + self.phi_tmp @ D[:, i - 3]

        psi[:, 9, 9] = np.einsum('ijk, j, k->i', psi_tmp, B_1, B_2) + self.phi_tmp @ B_2

        return phi, psi

    def propagate_with_impulse(self, t_eval, X):
        return solve_ivp(
            self.dynamics,
            [t_eval[0], t_eval[-1]],
            X[:6],
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            method='DOP853',
            dense_output=True
        ).y.T[:-1]
        
    # Check Hessian and sensitivity matrix
    def calculate_U_and_Q(self, x_nom, observer_k, measurement_model, epsilon=1e-4):
        U = np.zeros((self.MEASUREMENT_DIM, 6))
        Q = np.zeros((self.MEASUREMENT_DIM, 6, 6))

        for i in range(6):  # Perturb each state component
            dx = np.zeros(6)
            dx[i] = epsilon

            z_plus = measurement_model(x_nom + dx, observer_k)
            z_minus = measurement_model(x_nom - dx, observer_k)

            U[:, i] = (z_plus - z_minus) / (2 * epsilon)

            for j in range(6):
                dxj = np.zeros(6)
                dxj[j] = epsilon

                z_pp = measurement_model(x_nom + dx + dxj, observer_k)
                z_pm = measurement_model(x_nom + dx - dxj, observer_k)
                z_mp = measurement_model(x_nom - dx + dxj, observer_k)
                z_mm = measurement_model(x_nom - dx - dxj, observer_k)

                Q[:, i, j] = (z_pp - z_pm - z_mp + z_mm) / (4 * epsilon**2)

        return U, Q


    def do_calc(self):      
        for i in range(self.i_max):
            expected_points = self.propagate_with_impulse(self.t_eval, self.X_i)

            for j in range(self.K):
                self.z_exp[j] = self.transform_state(expected_points[j], self.observer[j])

            # Propagate orbit and find modified STT
            phi, psi = self.compute_stt(self.t_eval, self.X_i, self.propagate_with_impulse)

            U = []
            Q = []

            for k in range(self.K):
                x_k = expected_points[k]
                observer_k = self.observer[k]
                U_k, Q_k = self.calculate_U_and_Q(x_k, observer_k, self.transform_state)
                U.append(U_k)
                Q.append(Q_k)

            U = np.array(U)
            Q = np.array(Q)

            Xi = np.einsum('ijk,kp->ijp', U, phi)
            Theta = np.einsum('ijk,kpq->ijpq', U, psi) + np.einsum('ijkl,kp,lq->ijpq', Q, phi, phi)
            Sigma = Theta.reshape(self.K * self.MEASUREMENT_DIM, self.PARAM_DIM, self.PARAM_DIM)
            Omega = Xi.reshape(self.K * self.MEASUREMENT_DIM, self.PARAM_DIM)

            W = np.zeros((self.K, self.K, self.MEASUREMENT_DIM, self.MEASUREMENT_DIM))
            for i in range(self.K):
                m = 0.5 * np.einsum('iab,ab->i', Theta[i, :, :], self.P_i)

                term1 = np.einsum('jpq,ab,pq->jab', Theta[i, :, :], self.P_i, self.P_i)
                term2 = np.einsum('jpq,ap,bq->jab', Theta[i, :, :], self.P_i, self.P_i)
                term3 = np.einsum('jpq,aq,bp->jab', Theta[i, :, :], self.P_i, self.P_i)
                P_global = np.einsum('ia, jq, aq -> ij', Xi[i, :], Xi[i, :], self.P_i) - np.outer(m, m) + 0.25 * np.einsum('iab,jab->ij', (term1 + term2 + term3), Theta[i, :, :])

                W[i, i, :, :] = P_global + self.R
            
            W = np.linalg.pinv(W)
            W_norm = W / np.linalg.norm(W)
            W_norm = W_norm.reshape((self.K * self.MEASUREMENT_DIM, self.K * self.MEASUREMENT_DIM))
            delta_z = (self.z_tau - self.z_exp).reshape(self.K * self.MEASUREMENT_DIM)

            H = Omega.T @ W_norm @ Omega
            b = Omega.T @ W_norm @ delta_z
            delta_X_linear = np.linalg.pinv(H) @ b

            Gamma_bar = Omega + np.einsum('ikp,p->ik', Sigma, delta_X_linear)
            delta_Z_linear = -2 * Gamma_bar.T @ W_norm @ delta_z
            Omega_linear = -2 * Gamma_bar.T @ W_norm @ Omega
            Sigma_linear = -2 * np.einsum('ji,jk,kpq->ipq', Gamma_bar, W_norm, Sigma)

            # correction = 0.5 * np.einsum('ijk,j,k->i', Sigma_linear, delta_X_linear, delta_X_linear)
            # delta_X_hat = delta_X_linear - np.linalg.solve(Omega, correction)

            H_inv = np.linalg.pinv(H)
            tmp = np.linalg.pinv(Omega_linear) @ delta_Z_linear
            P_dx = np.linalg.pinv(Gamma_bar.T @ W_norm @ Omega) @ Gamma_bar.T @ W_norm @ self.R_2 @ W_norm.T @ Gamma_bar @ np.linalg.pinv(Omega.T @ W_norm @ Gamma_bar)
            delta_X_hat = tmp - 0.5 * np.linalg.pinv(Omega_linear) @ np.einsum('mij,i,j->m', Sigma_linear, tmp, tmp)

            # Update estimate
            self.X_i = self.X_i + delta_X_linear
            self.P_i = P_dx

            if self.record:
                tmp_result = [i, delta_X_linear]
                diff = self.X_0 - self.X_i
                for l in range(self.MEASUREMENT_DIM):
                    tmp_result.append(diff[l])

                self.results.append(tmp_result)

            #Convergence check
            if np.linalg.norm(delta_X_linear) <= self.nu:
                print("Successfully converged!")

                for j in self.results:
                    print(j)

                print(self.z_tau - self.z_exp)
                break

    def plotting(self):
        initial_guess = self.propagate_with_impulse(self.t_eval, self.X_0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.target[:, 0], self.target[:, 1], self.target[:, 2], color="red", label="Actual Orbit")
        ax.plot(initial_guess[:, 0], initial_guess[:, 1], initial_guess[:, 2], color="black", label = "Initial Guess")
        ax.plot(self.expected_points[:, 0], self.expected_points[:, 1], self.expected_points[:, 2], color="blue", label = "Converged Guess")
        ax.set_title("Orbital Trajectory with Impulse")
        plt.legend(loc="upper left")
        plt.show()