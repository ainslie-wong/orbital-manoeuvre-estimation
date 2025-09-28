import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import time_range
from poliastro.maneuver import Maneuver
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
mu_e    = 398600.44    # km**3/2**2
R_e     = 6378.137      # km
J2      = 1.08264e-3

PARAM_DIM = 10
STATE_DIM = 6

def transform_state(x_target, x_observer):
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


def generate_input_points(a, e, i, raan, argp, nu, noise, dvx, dvy, dvz, man_time, t_eval):
    # Sample orbit test
    epoch = Time("2025-01-01 00:00:00", scale="utc")
    a = (a + R_e) * u.km
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
        orbit = orbit.propagate(100 * u.s)
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

def dynamics(t, state):
    r = state[:3]
    v = state[3:]

    # J2 Perturbation
    a_j2r = [mu_e * r[0] * J2 * R_e**2/np.linalg.norm(r)**5 * (-3/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2),
            mu_e * r[1] * J2 * R_e**2/np.linalg.norm(r)**5 * (-3/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2),
            mu_e * r[2] * J2 * R_e**2/np.linalg.norm(r)**5 * (-9/2 + 15/2 * r[2]**2/np.linalg.norm(r)**2)
            ]
    
    norm_r = np.linalg.norm(r)
    a = -mu_e/norm_r**3 * r + a_j2r 
    return np.concatenate([v, a])


def propagate_with_impulse(t_eval, X):
    r0 = X[:3]
    v0 = X[3:6]
    dv = X[6:9]
    t_impulse = X[9]
    t_eval = np.array(t_eval)

    # Separate times before and after impulse
    t_pre = t_eval[t_eval <= t_impulse]
    t_post = t_eval[t_eval > t_impulse]

    if len(t_pre) == 1:
        t_pre = np.hstack((t_pre, t_impulse))

    # Propagate before impulse
    if len(t_pre) > 0:
        sol_pre = solve_ivp(
            dynamics,
            [t_pre[0], t_pre[-1]],
            np.concatenate([r0, v0]),
            t_eval=t_pre,
            rtol=1e-8,
            atol=1e-10,
            method='DOP853'
        )
        state_at_impulse = sol_pre.y[:, -1]
    else:
        state_at_impulse = np.concatenate([r0, v0])
        sol_pre = None

    # Apply impulse (Δv)
    state_after_impulse = np.concatenate([state_at_impulse[:3], state_at_impulse[3:] + dv])

    if sol_pre:
        sol_pre.y[:, -1] = state_after_impulse

    # Propagate after impulse
    if len(t_post) > 0:
        sol_post = solve_ivp(
            dynamics,
            [t_post[0], t_post[-1]],
            state_after_impulse,
            t_eval=t_post,
            rtol=1e-8,
            atol=1e-10,
            method='DOP853'
        )
    else:
        sol_post = None

    # Stitch the two segments
    trajectory = []
    if sol_pre and len(sol_pre.y) > 0:
        trajectory.extend(sol_pre.y.T)
    if sol_post and len(sol_post.y) > 0:
        trajectory.extend(sol_post.y.T[1:])

    return np.array(trajectory)


t_eval = np.arange(0, 1900, 100)
observer = generate_input_points(500, 0.01, 45.05, 29.93, 132.9, -107.74, 0, 0, 0, 0, 0, t_eval)
target = generate_input_points(1000, 0.02, 45, 94.80, 199.00, -54.13, 0, 1000, 1000, 1000, 900, t_eval)
print("I have generated the observer and target points")
# Set initial estimation and covariance
X_0 = np.hstack(([-2.41989719e+03, -6.23692850e+03,  2.93250127e+03,  4.72419828e+00,
 -3.72951031e+00, -4.39875044e+00  1.34118013e-02  7.54235999e-03
 -1.00549621e-02  9.10000039e+02]))
expected_points = propagate_with_impulse(t_eval, X_0)
print(target)
print("These are the shapes", target.shape, expected_points.shape)
print(expected_points)
#initial_guess = propagate_with_impulse(t_eval, np.hstack((target[0], np.array([1000/1000, 1000/1000, 1000/1000, 880]))))


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(target[:, 0], target[:, 1], target[:, 2], color="red")
ax.plot(expected_points[:, 0], expected_points[:, 1], expected_points[:, 2], color="blue")
#ax.plot(second_guess[:,0], second_guess[:, 1], second_guess[:, 2], color="green")
ax.set_title("Orbital Trajectory with Impulse")
plt.show()