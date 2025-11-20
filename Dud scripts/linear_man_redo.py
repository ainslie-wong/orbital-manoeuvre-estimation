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
    i_max = 1  # Iteration limit

    P_0 = np.diag([100**2, 100**2, 100**2, 1e-3**2, 1e-3**2, 1e-3**2, 5e-3**2, 5e-3**2, 5e-3**2, 10**2])
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

        for i in range(self.K):
            self.z_tau[i] = self.transform_state(self.target[i], self.observer[i])

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
        v = state[3:self.STATE_DIM]

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
        return states

    #y_ref first differential
    def calc_first_diff(self, x, t):
        r = x[:3]
        v = x[3:self.STATE_DIM]
        epsilon = 1e-6 * max(1.0, np.linalg.norm(r))

        dadr = np.zeros((3, 3))

        for i in range(3):
            dr = np.zeros(3)
            dr[i] = epsilon

            state_plus = np.concatenate([r + dr, v])
            state_minus = np.concatenate([r - dr, v])

            a_plus = self.dynamics(0, state_plus)[3:]
            a_minus = self.dynamics(0, state_minus)[3:]
            dadr[:, i] = (a_plus - a_minus) / (2 * epsilon)

        A = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [dadr, np.zeros((3, 3))]
        ])

        return A

    # used to solve for phi when integrating
    def combined_dynamics(self, t, y):
        x = y[:self.STATE_DIM]
        phi_flat = y[self.STATE_DIM:]
        phi = phi_flat.reshape((self.STATE_DIM, self.PARAM_DIM))

        dxdt = self.dynamics(t, x)
        A = self.calc_first_diff(x, t)
        dphidt = A @ phi

        return np.hstack((dxdt, dphidt.flatten()))


    def compute_stm(self):
        phi0 = np.zeros((self.STATE_DIM, self.PARAM_DIM))
        phi0[:, :self.STATE_DIM] = np.eye(self.STATE_DIM)
        phi0 = np.hstack((np.eye(6), np.zeros((6, 4)))).reshape(-1)
        #inside a loop to update X_i with each state in expected_points
        y0 = np.hstack((self.X_i[:self.STATE_DIM], phi0))
        sol = solve_ivp(self.combined_dynamics, [0, self.t_eval[-1]], y0, t_eval=self.t_eval, method='RK23')
        phi = sol.y[self.STATE_DIM:, :].T
        phi = phi.reshape((len(self.t_eval), self.STATE_DIM, self.PARAM_DIM))

        return phi

    def propagate(self, t_eval, X):
        r0 = X[:3]
        v0 = X[3:6]
        t_eval = np.array(t_eval)

        # Propagate before impulse
        sol = solve_ivp(
            self.dynamics,
            [t_eval[0], t_eval[-1]],
            np.concatenate([r0, v0]),
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-8,
            method='RK23',
            dense_output=True
        )

        return np.array(sol.y.T)
        

    def calculate_U(self, x_target, x_observer, measurement_model, epsilon = 1e-4):
        U = np.zeros((self.MEASUREMENT_DIM, self.STATE_DIM))

        for i in range(3):  # Perturb each state component
            dx = np.zeros(6)
            dx[i] = epsilon

            h_plus = measurement_model(x_target + dx, x_observer)
            h_minus = measurement_model(x_target - dx, x_observer)

            U[:, i] = (h_plus - h_minus) / (2 * epsilon)

        return U

    def do_calc(self):
        for i in range(self.i_max):
            self.expected_points = self.propagate(self.t_eval, self.X_i)

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
            Omega = Omega.reshape(self.K * self.MEASUREMENT_DIM, self.PARAM_DIM)
            
            # Calculate weight matrix and normalise
            W = np.zeros((self.K * self.MEASUREMENT_DIM, self.K * self.MEASUREMENT_DIM))
            for j in range(self.K):
                W[j * 2:j * 2 + 2, j * 2:j * 2 + 2] = np.linalg.pinv(Omega[j, :] @ self.P_i @ Omega[j, :].T + self.R_meas)

            W = W / np.linalg.norm(W)

            H = Omega.T @ W @ Omega
            b = Omega.T @ W @ delta_z
            delta_X_linear = np.linalg.pinv(H) @ b

            # Update state and covariance
            self.X_i = self.X_i + delta_X_linear

            P_dx = np.linalg.pinv(Omega.T @ W @ Omega) @ Omega.T @ W @ self.R_2 @ W.T @ Omega @ np.linalg.pinv(Omega.T @ W @ Omega)
            self.P_i = P_dx

            print("delta x:", delta_X_linear, "\n")
            print("delta x norm (convergence limit):\n", np.linalg.norm(delta_X_linear), "\n")

            # Convergence check
            if np.linalg.norm(delta_X_linear) <= self.nu:
                print("Successfully converged!")

                print("Difference in initial state to the iteration's estimated state")
                for j in self.results:
                    print(j)

                break
        print("Estimated State:", self.X_i)
        print("Original State:", self.X_0)

    def plotting(self):
        initial_guess = self.propagate(self.t_eval, self.X_0)
        self.expected_points = self.propagate(self.t_eval, self.X_i)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.target[:, 0], self.target[:, 1], self.target[:, 2], color="red", label="Actual Orbit")
        ax.plot(initial_guess[:, 0], initial_guess[:, 1], initial_guess[:, 2], color="black", label = "Initial Guess")
        ax.plot(self.expected_points[:, 0], self.expected_points[:, 1], self.expected_points[:, 2], color="blue", label = "Converged Guess")
        ax.set_title("Orbital Trajectory with Impulse")
        plt.legend(loc="upper left")
        plt.show()
