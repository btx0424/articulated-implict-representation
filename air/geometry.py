import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from flax import struct

class PyTreeNode(struct.PyTreeNode):

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)

    @property
    def at(self):
        return 


def index_set(tree, other, index):
    return jax.tree_map(lambda a, b: a.at[index].set(b), tree, other)


class Transform(PyTreeNode):
    pos: ArrayLike
    rot: ArrayLike

    @classmethod
    def zeros(cls, shape: int | tuple):
        if isinstance(shape, int):
            shape = (shape,)
        pos = jnp.zeros((*shape, 3))
        rot = jnp.zeros((*shape, 4)).at[..., 0].set(1.)
        return cls(pos=pos, rot=rot)
    
    @classmethod
    def create(cls, pos: ArrayLike=None, rot: ArrayLike=None):
        assert not (pos is None and rot is None)
        if pos is None:
            pos = jnp.zeros(rot.shape[:-1]+(3,))
        if rot is None:
            rot = jnp.zeros(pos.shape[:-1]+(4,)).at[..., 0].set(1.)
        return cls(pos=pos, rot=rot)


def apply_inv(x: ArrayLike, transform: Transform):
    shape = x.shape
    x = x.reshape(-1, 3) - transform.pos
    x = jax.vmap(quat_rotate_inv, (0, None))(x, transform.rot)
    return x.reshape(shape)


def compose(a: Transform, b: Transform):
    pos = a.pos + quat_rotate(b.pos, a.rot)
    rot = quat_mul(a.rot, b.rot)
    return Transform(pos, rot)


class DoF(PyTreeNode):
    dof_type: ArrayLike
    axis: ArrayLike
    pos: ArrayLike # the joint position


def dof_to_transform(dof: DoF):
    transform = jax.lax.switch(
        dof.dof_type,
        [
            lambda axis, pos: Transform.create(pos=normalize(axis) * pos),
            lambda axis, angle: Transform.create(rot=angle_axis_to_quat(angle, axis)),
        ],
        dof.axis, dof.pos
    )
    return transform


def normalize(vec: ArrayLike):
    return vec / jnp.linalg.norm(vec, axis=-1)


def quat_conj(quat: ArrayLike):
    return quat * jnp.array([1., -1., -1., -1.])


def quat_mul(u: ArrayLike, v: ArrayLike):
    """Multiplies two quaternions.

    Args:
        u: (4,) quaternion (w,x,y,z)
        v: (4,) quaternion (w,x,y,z)

    Returns:
        A quaternion u * v.
    """
    return jnp.array([
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    ])


def quat_rotate(vec: ArrayLike, quat: ArrayLike):
    """Rotates a vector vec by a unit quaternion quat.

    Args:
        vec: (3,) a vector
        quat: (4,) a quaternion

    Returns:
        ndarray(3) containing vec rotated by quat.
    """
    if len(vec.shape) != 1:
        raise ValueError('vec must have no batch dimensions.')
    s, u = quat[0], quat[1:]
    r = 2 * (jnp.dot(u, vec) * u) + (s * s - jnp.dot(u, u)) * vec
    r = r + 2 * s * jnp.cross(u, vec)
    return r


def quat_rotate_inv(vec: ArrayLike, quat: ArrayLike):
    """Inverse-rotates a vector vec by a unit quaternion quat.

    Args:
        vec: (3,) a vector
        quat: (4,) a quaternion

    Returns:
        ndarray(3) containing vec rotated by quat.
    """
    return quat_rotate(vec, quat_conj(quat))


def euler_to_quat(rpy: ArrayLike):
    """
    
    Args:
        rpy: (3,) a rotation in euler angles
    
    Returns:
        ndarray (4,) the cooresponding quaternion
    """
    r, p, y = rpy
    cy = jnp.cos(y * 0.5)
    sy = jnp.sin(y * 0.5)
    cp = jnp.cos(p * 0.5)
    sp = jnp.sin(p * 0.5)
    cr = jnp.cos(r * 0.5)
    sr = jnp.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quat = jnp.array([qw, qx, qy, qz])

    return quat


def angle_axis_to_quat(angle: ArrayLike, axis: ArrayLike):
    theta = angle / 2
    x, y, z = normalize(axis) * jnp.sin(theta)
    w = jnp.cos(theta)
    return jnp.array([w, x, y, z])


# def kinematic_tree(
#     pos: ArrayLike, 
#     rot: ArrayLike,
#     # dof: ArrayLike,
#     parent_index: ArrayLike
# ):
#     def func(carry, i):
#         pos, rot = carry
#         rot_parent = rot[parent_index[i]]
#         pos = pos.at[i].set(
#             pos[parent_index[i]] + quat_rotate(pos[i], rot_parent)
#         )
#         rot = rot.at[i].set(
#             quat_mul(rot_parent, rot[i])
#         )

#         return (pos, rot), None
#     (pos, rot), _ = jax.lax.scan(
#         func, (pos, rot), jnp.arange(1, len(parent_index))
#     )
#     return pos, rot

def kinematic_tree(
    local_transform: Transform,
    dof: DoF,
    parent_index: ArrayLike
) -> Transform:
    dof_transform = jax.vmap(dof_to_transform)(dof)
    def func(transform, i):
        t = compose(transform[parent_index[i]], transform[i])
        t = compose(t, dof_transform[i])
        transform = index_set(
            transform, t, i
        )
        return transform, None
    transform, _ = jax.lax.scan(
        func, local_transform, jnp.arange(1, len(parent_index))
    )
    return transform
