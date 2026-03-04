from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _apply_step_bounds(step: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Scale a step so all entries fall within [lower, upper].
    If any entry exceeds upper (e.g., > 0), we zero the step (scale=0).
    Otherwise scale down to satisfy lower bound.
    """
    step = np.asarray(step, dtype=float)
    max_step = float(np.max(step)) if step.size else 0.0
    min_step = float(np.min(step)) if step.size else 0.0
    scale = 1.0
    if max_step > upper and upper != 0.0:
        scale = min(scale, upper / max_step)
    if min_step < lower and lower != 0.0:
        scale = min(scale, lower / min_step)  # both negative => positive scale < 1
    return step * scale
from jax import grad, jacfwd, jacrev


def hessian(f):
    return jacfwd(jacrev(f))


def newton_update(theta, loss_fn, data):
    """
    Basic Newton update. loss_fn signature: loss_fn(theta, data) -> scalar
    """
    g = grad(loss_fn)(theta, data)
    H = hessian(lambda th: loss_fn(th, data))(theta)
    delta = jnp.linalg.solve(H, -g)
    return theta + delta


def loop_gradient_step(loop_x, loop_mean, loop_target, eta=1.0e-3, max_step=None):
    """
    Gradient step for loop parameter using residual = (loop_mean - loop_target).
    This matches maxent_loop sign conventions where more negative implies stronger attraction.
    """
    grad = loop_mean - loop_target
    step = eta * grad
    if max_step is not None:
        step = jnp.clip(step, -max_step, max_step)
    return loop_x + step


def tkl_gradient_step(interaction_matrix, tkl_resid_flat, eta=1.0e-3, max_step=None):
    """
    Gradient step for type-type interaction matrix using residuals (T_sim - T_exp),
    consistent with maxent_loop Newton update sign.
    """
    k = interaction_matrix.shape[0]
    iu = jnp.triu_indices(k)
    step = eta * tkl_resid_flat
    if max_step is not None:
        step = jnp.clip(step, -max_step, max_step)
    new_mat = jnp.array(interaction_matrix)
    new_mat = new_mat.at[iu].add(step)
    new_mat = jnp.triu(new_mat) + jnp.triu(new_mat, k=1).T
    return new_mat, step


def ic_gradient_step(lambda_ic, phi_resid, eta=1.0e-3, max_step=None):
    """
    Simple IC update for full phi: gradient step on lambda_IC using residuals
    (phi_sim - phi_target).
    """
    step = eta * jnp.asarray(phi_resid)
    if max_step is not None:
        step = jnp.clip(step, -max_step, max_step)
    lambda_new = jnp.asarray(lambda_ic) + step
    return np.asarray(lambda_new), np.asarray(step)


def tkl_newton_step(
    interaction_matrix,
    phi_sim_vec,
    phi_exp_vec,
    pi_pj_mean,
    damp=5e-7,
    step_bounds=None,
):
    """
    Damped Newton step for TKL using OpenMiChroM-style update:
      g = phi_exp - phi_sim
      B = PiPj_mean - outer(phi_sim, phi_sim)
      lambdas_new = pinv(B) @ g
      lambdas_final = lambdas_old - damp * lambdas_new
    """
    g = np.asarray(phi_exp_vec) - np.asarray(phi_sim_vec)
    phi_sim_np = np.asarray(phi_sim_vec)
    B = np.asarray(pi_pj_mean) - np.outer(phi_sim_np, phi_sim_np)
    B_pinv = np.linalg.pinv(B)
    lambdas_new = B_pinv @ g
    step = -damp * lambdas_new
    if step_bounds is not None:
        step = _apply_step_bounds(step, float(step_bounds[0]), float(step_bounds[1]))
    k = interaction_matrix.shape[0]
    iu = jnp.triu_indices(k)
    new_mat = np.array(interaction_matrix, dtype=float)
    new_mat[iu] += step
    new_mat = np.triu(new_mat) + np.triu(new_mat, k=1).T
    return new_mat, step


def ic_newton_step(params, phi_sim, phi_exp, pi_pj_mean, damp=3e-7, step_bounds=None):
    """
    Damped Newton step for IC using OpenMiChroM-style update:
      g = phi_exp - phi_sim
      B = PiPj_mean - outer(phi_sim, phi_sim)
      lambdas_new = pinv(B) @ g
      lambdas_final = lambdas_old - damp * lambdas_new
    """
    g = np.asarray(phi_exp) - np.asarray(phi_sim)
    phi_sim_np = np.asarray(phi_sim)
    B = np.asarray(pi_pj_mean) - np.outer(phi_sim_np, phi_sim_np)
    B_pinv = np.linalg.pinv(B)
    lambdas_new = B_pinv @ g
    step = -damp * lambdas_new
    if step_bounds is not None:
        step = _apply_step_bounds(step, float(step_bounds[0]), float(step_bounds[1]))
    params = params.copy()
    params["gamma1"] = float(params["gamma1"] + step[0])
    params["gamma2"] = float(params["gamma2"] + step[1])
    params["gamma3"] = float(params["gamma3"] + step[2])
    return params, step


def ic_newton_step_fullphi(lambda_ic, phi_sim, phi_exp, pi_pj_mean, damp=3e-7, step_bounds=None):
    """
    Damped Newton step for IC using full phi (lambda per genomic distance):
      g = phi_exp - phi_sim
      B = PiPj_mean - outer(phi_sim, phi_sim)
      lambdas_new = pinv(B) @ g
      lambda_next = lambda_old - damp * lambdas_new
    """
    g = np.asarray(phi_exp) - np.asarray(phi_sim)
    phi_sim_np = np.asarray(phi_sim)
    B = np.asarray(pi_pj_mean) - np.outer(phi_sim_np, phi_sim_np)
    B_pinv = np.linalg.pinv(B)
    lambdas_new = B_pinv @ g
    step = -damp * lambdas_new
    if step_bounds is not None:
        step = _apply_step_bounds(step, float(step_bounds[0]), float(step_bounds[1]))
    lambda_next = np.asarray(lambda_ic, dtype=float) + step
    return lambda_next, step


def ic_newton_step_projected(
    params,
    phi_sim,
    phi_exp,
    pi_pj_mean,
    d_init,
    damp=3e-7,
    step_bounds=None,
):
    """
    Full-phi Newton step projected onto (gamma1, gamma2, gamma3).
    We first compute a lambda(d) update from the full phi vector, then
    fit gamma1/2/3 to the updated lambda(d) via least squares.
    """
    g = np.asarray(phi_exp) - np.asarray(phi_sim)
    phi_sim_np = np.asarray(phi_sim)
    B = np.asarray(pi_pj_mean) - np.outer(phi_sim_np, phi_sim_np)
    B_pinv = np.linalg.pinv(B)
    delta_lambda = -damp * (B_pinv @ g)

    dmax = int(phi_sim.shape[0])
    d = np.arange(d_init, d_init + dmax, dtype=float)
    logd = np.log(d)
    A = np.stack([1.0 / logd, 1.0 / d, 1.0 / (d * d)], axis=1)

    gamma_current = np.array([params["gamma1"], params["gamma2"], params["gamma3"]], dtype=float)
    lambda_current = A @ gamma_current
    lambda_target = lambda_current + delta_lambda

    gamma_new, _, _, _ = np.linalg.lstsq(A, lambda_target, rcond=None)
    gamma_new = np.asarray(gamma_new, dtype=float)

    gamma_step = gamma_new - np.asarray([gamma_current[0], gamma_current[1], gamma_current[2]], dtype=float)
    if step_bounds is not None:
        gamma_step = _apply_step_bounds(gamma_step, float(step_bounds[0]), float(step_bounds[1]))
    gamma_new = gamma_current + gamma_step

    params = params.copy()
    params["gamma1"] = float(gamma_new[0])
    params["gamma2"] = float(gamma_new[1])
    params["gamma3"] = float(gamma_new[2])
    return params, gamma_step, np.asarray(delta_lambda)
