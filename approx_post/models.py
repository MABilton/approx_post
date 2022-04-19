import jax
import jax.numpy as jnp

def from_surrojax_gp(surrojax_gp, use_fwd=True):
    def jax_func(theta, d):
        x = jnp.concatenate([theta, d], axis=-1)
        if x.ndim < 2:
            x = x[None,:]
        return surrojax_gp.predict(x, return_var=False)['mean']
    return from_jax(jax_func, use_fwd)

def from_jax(jax_func, use_fwd=True):
    model_grad = jax.jacfwd(jax_func, argnums=0) if use_fwd else jax.jacrev(jax_func, argnums=0)
    model_funcs = [jax_func, model_grad]
    for idx, func in enumerate(model_funcs):
        # Vectorise over batch dimension of theta and d, and over sample dimension of theta:
        model_funcs[idx] = jax.vmap(jax.vmap(func, in_axes=(0,None)), in_axes=(0,0))
    return tuple(model_funcs)