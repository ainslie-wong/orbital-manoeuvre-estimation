import numpy as np
from scipy.integrate import solve_ivp
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import time_range
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

PARAM_DIM = 10
STATE_DIM = 6
t_eval = np.arange(0, 1810, 10)
mu_e    = 398600.44    # km**3/2**2
R_e     = 6378.137      # km
J2      = 1.08264e-3

def compute_stt(t_eval, X, propagate_with_impulse, epsilon_vector = None):
    if epsilon_vector is None:
        epsilon_vector = np.array([10, 10, 10, 1, 1, 1, 5e-3, 5e-3, 5e-3, 1])  # r/v, Δv, t_impulse

    phi = np.zeros((STATE_DIM, PARAM_DIM))
    psi = np.zeros((STATE_DIM, PARAM_DIM, PARAM_DIM))
    phi_tmp = np.zeros((STATE_DIM, STATE_DIM))
    psi_tmp = np.zeros((STATE_DIM, STATE_DIM, STATE_DIM))
    A_1 = np.zeros((STATE_DIM, STATE_DIM))
    A_2 = np.zeros((STATE_DIM, STATE_DIM, STATE_DIM))
    I = np.identity(6)
    
    t_pre = t_eval[t_eval <= X[9]]
    t_post = t_eval[t_eval > X[9]]

    # Phi from t_0 - t (ie A_1)
    for j in range(STATE_DIM):
        epsilon_current = np.zeros((PARAM_DIM))
        epsilon_current[j] = epsilon_vector[j]
    
        x_plus_pre = propagate_with_impulse(t_pre, X + epsilon_current)[-1]
        x_minus_pre = propagate_with_impulse(t_pre, X - epsilon_current)[-1]
        A_1[:, j] = (x_plus_pre - x_minus_pre) / (2 * epsilon_vector[j])
    print(np.linalg.norm(A_1))
    # Psi from t_0 - t (ie A_2)
    for j in range(STATE_DIM):
        for k in range(STATE_DIM):
            X_pp = X.copy()
            X_pp[j] += epsilon_vector[j]
            X_pp[k] += epsilon_vector[k]

            X_pm = X.copy()
            X_pm[j] += epsilon_vector[j]
            X_pm[k] -= epsilon_vector[k]

            X_mp = X.copy()
            X_mp[j] -= epsilon_vector[j]
            X_mp[k] += epsilon_vector[k]

            X_mm = X.copy()
            X_mm[j] -= epsilon_vector[j]
            X_mm[k] -= epsilon_vector[k]

            x_pp = propagate_with_impulse(t_pre, X_pp)[-1]
            x_pm = propagate_with_impulse(t_pre, X_pm)[-1]
            x_mp = propagate_with_impulse(t_pre, X_mp)[-1]
            x_mm = propagate_with_impulse(t_pre, X_mm)[-1]

            A_2[:, j, k] = (x_pp - x_pm - x_mp + x_mm) / (4 * epsilon_vector[j] * epsilon_vector[k])

    # t - t_2
    X_no_dv = np.hstack((X[:6], 0, 0, 0, 1800))
    X_t1 = np.hstack((propagate_with_impulse(t_pre, X)[-1], X[6:]))
    X_no_man = np.hstack((propagate_with_impulse(t_pre, X_no_dv)[-1], X[6:]))

    # f
    f_plus = propagate_with_impulse(t_post, X_no_man)[-1]
    f_minus = propagate_with_impulse(t_post, X_t1)[-1]

    # g
    g_minus_p = propagate_with_impulse(np.hstack(([t_post[0] + epsilon_vector[9]], t_post[1:])), X_no_man)[-1]
    g_minus_m = propagate_with_impulse(np.hstack(([t_post[0] - epsilon_vector[9]], t_post[1:])), X_no_man)[-1]
    g_minus = (g_minus_p - g_minus_m) / (2 * epsilon_vector[9])

    g_plus_p = propagate_with_impulse(np.hstack(([t_post[0] + epsilon_vector[9]], t_post[1:])), X_t1)[-1]
    g_plus_m = propagate_with_impulse(np.hstack(([t_post[0] - epsilon_vector[9]], t_post[1:])), X_t1)[-1]
    g_plus = (g_plus_p - g_plus_m) / (2 * epsilon_vector[9])

    # H
    H_plus = np.zeros((STATE_DIM, STATE_DIM))
    H_minus = np.zeros((STATE_DIM, STATE_DIM))

    for j in range(STATE_DIM):
        x_plus_p = propagate_with_impulse(t_pre, X_t1 + epsilon_vector)[-1]
        x_minus_p = propagate_with_impulse(t_pre, X_t1 - epsilon_vector)[-1]
        H_plus[:, j] = (x_plus_p - x_minus_p) / (2 * epsilon_vector[:6])

        x_plus_m = propagate_with_impulse(t_pre, X_no_man + epsilon_vector)[-1]
        x_minus_m = propagate_with_impulse(t_pre, X_no_man - epsilon_vector)[-1]
        H_minus[:, j] = (x_plus_m - x_minus_m) / (2 * epsilon_vector[:6])
    
    B_1 =  f_minus - f_plus
    B_2 = g_minus - g_plus + 2 * np.einsum('kp,p ->k', H_plus, f_plus)
    C = H_minus - np.einsum('kl,lp->kp', H_plus, A_1)
    D = -H_plus

    # Phi and Psi for t1 - t2
    for j in range(STATE_DIM):
        epsilon_current = np.zeros((PARAM_DIM))
        epsilon_current[j] = epsilon_vector[j]

        x_plus_post = propagate_with_impulse(t_post, X_t1 + epsilon_current)[-1]
        x_minus_post = propagate_with_impulse(t_post, X_t1 - epsilon_current)[-1]
        phi_tmp[:, j] = (x_plus_post - x_minus_post) / (2 * epsilon_vector[j])
    
    for j in range(STATE_DIM):
        for k in range(STATE_DIM):
            X_t1pp = X.copy()
            X_t1pp[j] += epsilon_vector[j]
            X_t1pp[k] += epsilon_vector[k]

            X_t1pm = X.copy()
            X_t1pm[j] += epsilon_vector[j]
            X_t1pm[k] -= epsilon_vector[k]

            X_t1mp = X.copy()
            X_t1mp[j] -= epsilon_vector[j]
            X_t1mp[k] += epsilon_vector[k]

            X_t1mm = X.copy()
            X_t1mm[j] -= epsilon_vector[j]
            X_t1mm[k] -= epsilon_vector[k]

            x_pp = propagate_with_impulse(t_post, X_t1pp)[-1]
            x_pm = propagate_with_impulse(t_post, X_t1pm)[-1]
            x_mp = propagate_with_impulse(t_post, X_t1mp)[-1]
            x_mm = propagate_with_impulse(t_post, X_t1mm)[-1]

            psi_tmp[:, j, k] = (x_pp - x_pm - x_mp + x_mm) / (4 * epsilon_vector[j] * epsilon_vector[k])

    # Compile matrices
    # Phi    for i in range(STATE_DIM):
    for i in range(STATE_DIM):
        phi[:, i] = phi_tmp @ A_1[:, i]
    phi[:, 6:9] = phi_tmp[:, 3:6]
    phi[:, 9] = phi_tmp @ B_1

    # Psi
    psi[:, 6:9, 6:9] = psi_tmp[:, 3:6, 3:6]

    # 0 - 6 x 0 - 6
    for i in range(STATE_DIM):
        for j in range(STATE_DIM):
            psi[:, i, j] = np.einsum('ijk, j, k->i', psi_tmp, A_1[:, i], A_1[:, j]) + phi_tmp @ A_2[:, i, j]

        for j in range(6, 9):
            psi[:, i, j - 3] = np.einsum('ijk, j, k->i', psi_tmp, A_1[:, i], I[:, j - 3])
            psi[:, j, i - 3] = np.einsum('ijk, j, k->i', psi_tmp, A_1[:, i], I[:, j - 3])

        psi[:, i, 9] = np.einsum('ijk, j, k->i', psi_tmp, A_1[:, i], B_1) + phi_tmp @ C[:, i]
        psi[:, 9, i] = np.einsum('ijk, j, k->i', psi_tmp, A_1[:, i], B_1) + phi_tmp @ C[:, i]
    
    for i in range(6, 9):
        psi[:, i, 9] = psi_tmp[:, :, i - 3] @ B_1 + phi_tmp @ D[:, i - 3]
        psi[:, 9, i] = psi_tmp[:, :, i - 3] @ B_1 + phi_tmp @ D[:, i - 3]

    psi[:, 9, 9] = np.einsum('ijk, j, k->i', psi_tmp, B_1, B_2) + phi_tmp @ B_2
    print(np.linalg.norm(psi), np.linalg.norm(phi))
    return phi, psi

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
        trajectory.extend(sol_post.y.T)

    return np.array(trajectory)


# Propagate orbit and find modified STT
X_i = np.hstack(np.array([-2.40989718e+03, -6.22692852e+03,  2.92250128,  4.70955888,
 -3.73993346, -4.38007974, 10/1000, 2/1000, 5/1000, 880]))
phi, psi = compute_stt(t_eval, X_i, propagate_with_impulse)
print(phi, psi)