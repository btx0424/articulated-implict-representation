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

class Decoder(nn.Module):
    num_layers: int
    num_units: int

    @nn.compact
    def __call__(self, x):

        f = nn.Sequential([
            nn.Sequential([nn.Dense(self.num_units), nn.elu])
            for _ in range(self.num_layers)
        ])(x)
        f = nn.Dense(1)(f)

        return f

class LatentDecoder(nn.Module):
    num_layers: int
    num_units: int

    @nn.compact
    def __call__(self, latent, x):

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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B = jax.random.normal(key, (128, 3))

    def pos_encoding(x):
        x_proj = (2*jnp.pi*x) @ B.T
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

    # embed = nn.Embed(9, 64)
    # code = embed.init(key, jnp.arange(9))
    code_dim = 128
    code = jax.random.normal(key, (128,))

    decoder = Decoder(4, 256)
    params = decoder.init(key, pos_encoding(jnp.zeros((1, 3))))
    train_state = TrainState.create(
        apply_fn=decoder.apply,
        params=params,
        tx=optax.adam(1e-4)
    )    
    
    decoder = LatentDecoder(4, 256)
    params = decoder.init(
        key, jnp.zeros((1, code_dim)), pos_encoding(jnp.zeros((1, 3)))
    )
    train_state = TrainState.create(
        apply_fn=decoder.apply,
        params=(params, code),
        tx=optax.adam(1e-4)
    ) 

    
    @jax.value_and_grad
    def loss_fn(params_and_code, x, y):
        params, code = params_and_code
        pred = jax.vmap(decoder.apply, (None, None, 0))(params, code, x)
        loss = 0.5 * jnp.mean((pred-y)**2)
        return loss

    @jax.jit
    def update_fn(train_state: TrainState, x, y):
        loss, grads = loss_fn(train_state.params, x, y)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    path = os.path.join(os.path.dirname(__file__), "ant.npy")
    logging.info(path)

    ant = jnp.load(path)
    x = jnp.linspace(-1, 1, 72)
    x = jnp.stack(jnp.meshgrid(x, x, x), axis=-1)
    x, y = jax.tree_map(flatten, (x, ant[..., None]))
    x = pos_encoding(x)

    for i in range(100):
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(key, len(x)).reshape(32, -1)
        # idx = jnp.arange(len(x)).reshape(128, -1)
        for batch_idx in tqdm(idx):
            train_state, loss = update_fn(train_state, x[batch_idx], y[batch_idx])
        
        psnr = -10 * jnp.log10(2*loss)
        print(loss.item(), psnr.item())
    
    import mcubes
    import matplotlib.pyplot as plt

    shape = decoder.apply(train_state.params, x).reshape(ant.shape)
    vertices, triangles = mcubes.marching_cubes(np.asarray(shape), 0)
    fig = plt.figure()
    ax = plt.subplot(projection="3d")
    ax.plot_trisurf(*vertices.T, triangles=triangles)
    ax.axis("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.savefig("vis.png")
