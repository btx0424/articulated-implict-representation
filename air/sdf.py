import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

def sdf_sphere(p: ArrayLike, r):
    return jnp.linalg.norm(p, axis=-1) - r


def sdf_capsule(p: ArrayLike, h, r):
    return jnp.linalg.norm(
        p.at[..., 0].set(p[..., 0]-jnp.clip(p[..., 0], 0., h)),
        axis=-1
    ) - r

