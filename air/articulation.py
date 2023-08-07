import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .geometry import PyTreeNode, Transform, DoF, euler_to_quat, kinematic_tree, apply_inv
from .sdf import sdf_sphere, sdf_capsule

class Ant(PyTreeNode):
    transform: Transform
    dof: DoF
    parent_idx: ArrayLike

    leg_len: float

    @classmethod
    def create(cls):
        rot = (
            jnp.zeros((9, 3))
            .at[1:5, 2].set(jnp.linspace(0, 2, 5)[:4] * jnp.pi + jnp.pi/4)
        )
        rot = jax.vmap(euler_to_quat)(rot)
        leg_len = 0.2 * jnp.sqrt(2)
        pos = jnp.array([
            [0., 0., 0.],
            [0.2, 0.2, 0.],
            [-0.2, 0.2, 0.],
            [-0.2, -0.2, 0.],
            [0.2, -0.2, 0.],
            ################
            [leg_len, 0., 0.],
            [leg_len, 0., 0.],
            [leg_len, 0., 0.],
            [leg_len, 0., 0.],
        ])
        dof = DoF(
            jnp.array([0, 1, 1, 1, 1, 1, 1, 1, 1]),
            jnp.array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
            ]),
            jnp.zeros(9)
            .at[1:5].set(-jnp.pi/6)
            .at[5:9].set(jnp.pi/3)
        )

        return cls(
            transform=Transform(pos, rot),
            dof=dof,
            parent_idx=jnp.array([0, 0, 0, 0, 0, 1, 2, 3, 4]),
            leg_len=leg_len
        )


    @staticmethod
    def sdf(ant: "Ant", x: ArrayLike):
        shape = x.shape[:-1]
        transform = kinematic_tree(ant.transform, ant.dof, ant.parent_idx)
        x = jax.vmap(apply_inv, (None, 0))(x.reshape(-1, 3), transform)
        leg_len = ant.leg_len
        sdfs = []
        sdfs.append(sdf_sphere(x[0], 0.25))
        sdfs.append(sdf_capsule(x[1], leg_len, 0.08))
        sdfs.append(sdf_capsule(x[2], leg_len, 0.08))
        sdfs.append(sdf_capsule(x[3], leg_len, 0.08))
        sdfs.append(sdf_capsule(x[4], leg_len, 0.08))
        sdfs.append(sdf_capsule(x[5], leg_len, 0.08))
        sdfs.append(sdf_capsule(x[6], leg_len, 0.08))
        sdfs.append(sdf_capsule(x[7], leg_len, 0.08))
        sdfs.append(sdf_capsule(x[8], leg_len, 0.08))
        sdf = jnp.stack(sdfs, axis=-1).min(axis=-1)
        return sdf.reshape(shape)


from typing import Callable
def decode_sdf(
    x: ArrayLike,
    sdf_fn: Callable[[ArrayLike], ArrayLike],
    transform: Transform, 
    dof: DoF, 
    parent_index: ArrayLike,
    sdf_fn_args = ()
):
    """
    Args:
        x (*, 3): The input coordinates.
        
    Returns:

    """
    if len(sdf_fn_args) == 0:
        sdf_fn_args = (None for _ in range(len(parent_index)))
    
    assert parent_index.ndim == 1
    assert len(parent_index) == len(sdf_fn_args)

    transform = kinematic_tree(transform, dof, parent_index)
    d = compose_sdf(x, sdf_fn, transform, sdf_fn_args)
    return d


def compose_sdf(
    x: ArrayLike,
    sdf_fn: Callable,
    transform: Transform,
    sdf_fn_args=(),
):
    if len(sdf_fn_args) == 0:
        sdf_fn_args = (None for _ in range(len(transform.pos)))
    
    batch_shape = x.shape[:-1]
    def func(carry, input):
        transform, arg = input
        x_ = apply_inv(x, transform)
        d = sdf_fn(x_, arg)
        carry = jnp.minimum(carry, d)
        return carry, ()

    d, _ = jax.lax.scan(
        func,
        jnp.full((*batch_shape, 1), jnp.inf),
        (transform, sdf_fn_args)
    )
    return d.reshape(batch_shape)