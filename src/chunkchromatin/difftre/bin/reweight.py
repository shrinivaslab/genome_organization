from __future__ import annotations

import jax.numpy as jnp


def compute_weights(delta_energy: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Compute normalized reweighting factors.
    delta_energy = U_theta - U_ref (lower updated energy => higher weight)
    """
    w = jnp.exp(-beta * delta_energy)
    w = w / jnp.sum(w)
    return w


def effective_sample_size(weights: jnp.ndarray) -> jnp.ndarray:
    """
    N_eff = exp(-sum(w * log w))
    """
    return jnp.exp(-jnp.sum(weights * jnp.log(weights)))
