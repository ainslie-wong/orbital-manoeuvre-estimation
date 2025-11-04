import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import matplotlib.pyplot as plt

class PropagateSatelliteLinearMan():
    # Constants
    mu_e    = 398600.44    # km**3/s**2
    R_e     = 6378.137      # km
    J2      = 1.08264e-3

    PARAM_DIM = 10
    STATE_DIM = 6
    MEASUREMENT_DIM = 2
    nu = 1e-3   # Convergence limit
    i_max = 10  # Iteration limit

    P_0 = np.diag([100**2, 100**2, 100**2, 1e-2**2, 1e-2**2, 1e-2**2, 5e-3**2, 5e-3**2, 5e-3**2, 50**2])
    sigma_noise = 1e-4 # Noise during simulation point generation
    R_meas = sigma_noise**2 * np.eye(MEASUREMENT_DIM)   # 2 x 2 measurement covariance matrix

    results = []

    def __init__(self, num_measurement, time_split, x_dv, y_dv, z_dv, t_dv, xi_dv, yi_dv, zi_dv, ti_dv, record = False):
        self.record = record
        self.time_split = time_split

        self.K = num_measurement
        self.t_eval = np.arange(0, (num_measurement + 1)* time_split, time_split)

        self.observer = self.generate_input_points(500, 0.01, 45.05, 29.93, 132.9, -107.74, 1e-5, 0, 0, 0, 0, self.t_eval)
        self.target = self.generate_input_points(1000, 0.02, 45, 94.80, 199.00, -54.13, 1e-5, x_dv, y_dv, z_dv, t_dv, self.t_eval)
        self.X_0 = np.hstack((self.target[0] + [10, 10, 10, 5e-3, 5e-3, 5e-3], np.array([xi_dv/1000, yi_dv/1000, zi_dv/1000, ti_dv])))
        print("I have generated the observer and target points")

        self.X_i = self.X_0
        self.P_i = self.P_0

        self.z_tau = np.zeros((self.K, 2))
        self.z_exp = np.zeros((self.K, 2))

        for i in range(self.K - 1):
            self.z_tau[i] = self.transform_state(self.target[i], self.observer[i])

        self.record_to_file("initial_man", self.z_exp - self.z_tau)

        print("I have generated the expected points")

        # Covariance matrix
        self.R_2 = self.sigma_noise**2 * np.eye(self.K * self.MEASUREMENT_DIM)

    # Turns r, v state vector into azimuth/elevation
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

    # J2 dynamics-inclusive dynamics for spacecraft
    def dynamics(self, t, state):
        r = state[:3]
        v = state[3:]

        # J2 Perturbation
        a_j2r = [self.mu_e * r[0] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-3/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2),
                self.mu_e * r[1] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-3/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2),
                self.mu_e * r[2] * self.J2 * self.R_e**2/np.linalg.norm(r)**5 * (-9/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2)
                ]
        
        norm_r = np.linalg.norm(r)
        a = -self.mu_e * r/norm_r**3 + a_j2r

        return np.concatenate([v, a])

    # Generates the simulated satellite state vectors at each measurement time using orbital elements (from poliastro library)
    def generate_input_points(self, a, e, i, raan, argp, nu, noise, dvx, dvy, dvz, man_time, t_eval):
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
            orbit = orbit.propagate(self.time_split * u.s)
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

    #y_ref first differential
    def calc_first_diff(self, x, t):
        epsilon = 1e-4
        r = x[:3]
        v = x[3:6]

        dadr = np.zeros((3, 3))

        for i in range(3):
            dr = np.zeros(3)
            dr[i] = epsilon

            state_plus = np.concatenate([r + dr, v])
            state_minus = np.concatenate([r - dr, v])

            a_plus = self.dynamics(t, state_plus)[3:]
            a_minus = self.dynamics(t, state_minus)[3:]

            dadr[:, i] = (a_plus - a_minus) / (2 * epsilon)

        A = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [dadr, np.zeros((3, 3))]
        ])

        return A

    # used to solve for phi when integrating
    def combined_dynamics_pre(self, t, y):
        x = y[:self.PARAM_DIM]
        phi_flat = y[self.PARAM_DIM:]
        phi = phi_flat.reshape((self.STATE_DIM, self.PARAM_DIM))

        dxdt = self.dynamics(t, x)
        A = self.calc_first_diff(x, t)
        dphidt = A @ phi

        return np.hstack((dxdt, dphidt.flatten()))


    # Eqn. 8, 10,27 and (attempted) 29
    def compute_stm(self):
        # Get the pre- and post- manoeuvre time evaluation, including the estimated manoeuvre time as pre and post for x plus and minus
        t_pre = self.t_eval[self.t_eval <= self.X_i[9]]
        if t_pre[-1] != self.X_i[9]:
            t_pre = np.hstack((t_pre, self.X_i[9]))

        t_post = [self.X_i[9]]
        for time in self.t_eval[self.t_eval > self.X_i[9]]:
            t_post.append(time)


        phi0 = np.zeros((self.STATE_DIM, self.PARAM_DIM))
        phi0[:, :6] = np.eye(self.STATE_DIM)

        #Calculate phi for t0 - t
        y0_pre = np.hstack((self.X_i, phi0.flatten()))

        sol = solve_ivp(self.combined_dynamics_pre, [0, self.t_eval[-1]], y0_pre, t_eval=self.t_eval, method='RK23')
        phi = sol.y[self.PARAM_DIM:, :-1].T
        phi = phi.reshape((self.K, self.STATE_DIM, self.PARAM_DIM))

        # State after impulse
        X_plus = np.hstack((self.propagate_with_impulse(t_pre, self.X_i)[-1], 0, 0, 0, 0))

        #State before impulse
        X_minus = np.array(X_plus - [0, 0, 0, self.X_i[6], self.X_i[7], self.X_i[8], 0, 0, 0, 0])
        
        # Calculate B = f- - f+
        f_minus = self.dynamics(self.X_i[9], X_minus[:6])
        f_plus = self.dynamics(self.X_i[9], X_plus[:6])
        B = f_minus - f_plus

        # Calculate phi for t - t2
        phi_post0 = phi[-1, :, :].copy()
        phi_post0[3:, 6:9] += np.eye(3)
        phi_post0[:, 9] = f_minus[:6]

        y0_post = np.hstack((X_plus, phi_post0.flatten()))
        sol = solve_ivp(self.combined_dynamics_pre, [t_post[0], t_post[-1]], y0_post, t_eval=t_post, method='RK23')
        phi1 = sol.y[self.PARAM_DIM:, 1:-1].T
        phi1 = phi1.reshape((len(t_post) - 2, self.STATE_DIM, self.PARAM_DIM))

        # Compiling the post-manoeuvre matrix in accordance with Eqn. 29
        count = 0 # Use this count to find the associated post-manoeuvre phi for the associated A_1 matrix
        for point in range(self.K):
            if point > self.X_i[9]/self.time_split:
                phi[point, :, :6] = np.einsum('ij, jm -> im', phi1[count, :, :6], phi[point, :, :6])
                phi[point, :, 6:9] = phi1[count, :, 6:9]
                phi[point, :, 9] = phi1[count, :, 9] * B[:6]

                count +=  1

        return phi

    def propagate_with_impulse(self, t_eval, X):
        r0 = X[:3]
        v0 = X[3:6]
        dv = X[6:9]
        t_impulse = X[9]
        t_eval = np.array(t_eval)

        # Separate times before and after impulse
        t_pre = self.t_eval[self.t_eval <= t_impulse]
        if t_pre[-1] !=  t_impulse:
            t_pre = np.hstack((t_pre, t_impulse))

        t_post = [t_impulse]
        for time in self.t_eval[self.t_eval > t_impulse]:
            t_post.append(time)

        if len(t_pre) == 1:
            t_pre = np.hstack((t_pre, t_impulse))

        # Propagate before impulse
        if len(t_pre) > 0:
            sol_pre = solve_ivp(
                self.dynamics,
                [t_pre[0], t_pre[-1]],
                np.concatenate([r0, v0]),
                t_eval=t_pre,
                rtol=1e-6,
                atol=1e-8,
                method='RK23',
                dense_output=True
            )
            state_at_impulse = sol_pre.y[:, -1]
        else:
            state_at_impulse = np.concatenate([r0, v0])
            sol_pre = None

        # Apply impulse
        state_after_impulse = np.concatenate([state_at_impulse[:3], state_at_impulse[3:] + dv])

        if sol_pre:
            sol_pre.y[:, -1] = state_after_impulse

        # Propagate after impulse
        if len(t_post) > 0:
            sol_post = solve_ivp(
                self.dynamics,
                [t_post[0], t_post[-1]],
                state_after_impulse,
                t_eval=t_post,
                rtol=1e-8,
                atol=1e-10,
                method='RK23',
                dense_output=True
            )
        else:
            sol_post = None

        # Stitch the two segments
        trajectory = []
        if sol_pre and len(sol_pre.y) > 0:
            trajectory.extend(sol_pre.y[:, :-1].T)
        if sol_post and len(sol_post.y) > 0:
            trajectory.extend(sol_post.y[:, 1:-1].T)

        return np.array(trajectory)
        

    # From Eqn. 32
    def calculate_U(self, x_target, x_observer, measurement_model, epsilon = 1e-4):
        U = np.zeros((self.MEASUREMENT_DIM, self.STATE_DIM))

        for i in range(6):  # Perturb each state component
            dx = np.zeros(6)
            dx[i] = epsilon

            h_plus = measurement_model(x_target + dx, x_observer)
            h_minus = measurement_model(x_target - dx, x_observer)

            U[:, i] = (h_plus - h_minus) / (2 * epsilon)

        return U

    def do_calc(self):
        for i in range(self.i_max):
            self.expected_points = self.propagate_with_impulse(self.t_eval, self.X_i)

            for j in range(self.K - 1):
                self.z_exp[j] = self.transform_state(self.expected_points[j], self.observer[j])

            # Find STM (phi) and measurement Jacobian (U)
            phi = self.compute_stm()
            U = []

            for k in range(self.K):
                U_k = self.calculate_U(self.target[k], self.observer[k], self.transform_state)
                U.append(U_k)

            U = np.array(U)

            # Determine total sensitivity matrix at each measurement time
            Omega = np.zeros((self.K, self.MEASUREMENT_DIM, self.PARAM_DIM))
            for j in range(self.K):
                Omega[j, :] = U[j, :] @ phi[j, :]

            # Find difference in recorded and propagated azimuth/range measurements
            delta_z = (self.z_tau - self.z_exp).reshape((self.K * self.MEASUREMENT_DIM))
            Omega = Omega.reshape(self.K * self.MEASUREMENT_DIM, self.PARAM_DIM) # Eqn. 36 and 45

            # Calculate weight matrix and normalise
            # Eqn. 38, 39 and 50
            W = np.zeros((self.K * self.MEASUREMENT_DIM, self.K * self.MEASUREMENT_DIM))
            for i in range(self.K - 1):
                W[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = np.linalg.pinv(Omega[i, :] @ self.P_i @ Omega[i, :].T + self.R_meas)
            
            W = W / np.linalg.norm(W)

            H = Omega.T @ W @ Omega
            b = Omega.T @ W @ delta_z
            delta_X_linear = np.linalg.pinv(H) @ b # Eqn. 55

            # Update state and covariance
            self.X_i = self.X_i + delta_X_linear

            P_dx = np.linalg.pinv(Omega.T @ W @ Omega) @ Omega.T @ W @ self.R_2 @ W.T @ Omega @ np.linalg.pinv(Omega.T @ W @ Omega) #Eqn. 66
            self.P_i = P_dx

            # Record the iteration and difference in current estimated X_0 and actual initial guess
            if self.record:
                self.results.append([i, delta_X_linear, self.X_0 - self.X_i])

            print(delta_X_linear)
            print(np.linalg.norm(delta_X_linear))

            # Convergence check
            if np.linalg.norm(delta_X_linear) <= self.nu:
                print("Successfully converged!")

                for j in self.results:
                    print(j)

                #self.record_to_file("converged_man", self.z_exp - self.z_tau)

                break

        print(self.X_i)

    # Tmp function if feeling deluded and need to look at each propagation visually
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

    def record_to_file(self, filename, data):
        with open(filename, "w") as file:
            for line in data:
                file.write(f"{line}\n")