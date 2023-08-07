import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import flax.linen as nn
from flax.training.train_state import TrainState

import optax


from tqdm import tqdm
import os
import logging
from typing import Callable

class LatentDecoder(nn.Module):
    num_layers: int
    num_units: int
    pos_encoding: Callable

    @nn.compact
    def __call__(self, latent, x):
        x = self.pos_encoding(x)
        f = nn.Sequential([
            nn.Sequential([nn.Dense(self.num_units), nn.elu])
            for _ in range(self.num_layers)
        ])(jnp.concatenate([latent, x], axis=-1))
        f = nn.Dense(1)(f)

        return f


def flatten(x: ArrayLike, start_axis: int=0, end_axis: int=-2):
    end_axis = (end_axis + x.ndim) % x.ndim
    shape = (*x.shape[:start_axis], -1, *x.shape[end_axis+1:])
    return x.reshape(shape)


from air.articulation import compose_sdf
from air.geometry import kinematic_tree

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B = jax.random.normal(key, (128, 3))

    def pos_encoding(x):
        x_proj = (2*jnp.pi*x) @ B.T
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

    code_dim = 256
    code = jax.random.normal(key, (9, code_dim))
    
    decoder = LatentDecoder(5, 256, pos_encoding)
    params = decoder.init(
        key, jnp.zeros((1, code_dim)), jnp.zeros((1, 3))
    )
    

    train_state = TrainState.create(
        apply_fn=decoder.apply,
        params=(params, code),
        tx=optax.adam(3e-5)
    ) 

    @jax.value_and_grad
    def loss_fn(params_and_code, x, y):
        params, code = params_and_code
        sdf_fn = lambda x, code: jax.vmap(decoder.apply, (None, None, 0))(params, code, x)
        pred = compose_sdf(
            x=x,
            sdf_fn=sdf_fn,
            transform=transform,
            sdf_fn_args=code
        )
        loss = 0.5 * jnp.mean((pred-y)**2)
        return loss

    @jax.jit
    def update_fn(train_state: TrainState, x, y):
        loss, grads = loss_fn(train_state.params, x, y)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    path = os.path.join(os.path.dirname(__file__), "data/ant_0.pkl")
    logging.info(path)
    import pickle
    with open(path, "rb") as f:
        ant = pickle.load(f)
    
    transform = kinematic_tree(ant["transform"], ant["dof"], ant["parent_idx"])
    x = jnp.linspace(-1, 1, 72)
    x = jnp.stack(jnp.meshgrid(x, x, x), axis=-1)
    x, y = jax.tree_map(flatten, (x, ant["sdf"][..., None]))

    for i in range(100):
        key, subkey = jax.random.split(key)
        # idx = jax.random.permutation(key, len(x)).reshape(32, -1)
        idx = jnp.arange(len(x)).reshape(16, -1)
        for batch_idx in tqdm(idx):
            train_state, loss = update_fn(train_state, x[batch_idx], y[batch_idx])
        
        psnr = -10 * jnp.log10(2*loss)
        print(loss.item(), psnr.item())
    
    import mcubes
    import matplotlib.pyplot as plt

    params, code = train_state.params
    sdf_fn = lambda x, code: jax.vmap(decoder.apply, (None, None, 0))(params, code, x)
    shape = compose_sdf(
        x=x,
        sdf_fn=sdf_fn,
        transform=transform,
        sdf_fn_args=code
    ).reshape(72, 72, 72)
    vertices, triangles = mcubes.marching_cubes(np.asarray(shape), 0)
    print(vertices.shape)
    logging.info(vertices.shape)

    fig = plt.figure()
    ax = plt.subplot(projection="3d")
    ax.plot_trisurf(*vertices.T, triangles=triangles)
    ax.axis("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.savefig("vis.png")
