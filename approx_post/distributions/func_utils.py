import numpy as np
import jax
import jax.numpy as jnp

from .container import ArrayContainer

def create_grads(lp, transform=None):
    
    # Not using reparameterisation:
    if transform is None:
        # Compute gradients:
        lp_del_1 = jax.jacfwd(lp, argnums=0)
        lp_del_2 = jax.jacfwd(lp, argnums=1)
        transform_del_1 = jax.jacfwd(transform, argnums=1)
        # Vectorise functions:
        lp_del_1, lp_del_2, transform_del_1 = vectorise_functions(lp_del_1, lp_del_2, transform_del_1)
        return (lp_del_1, lp_del_2, transform_del_1)

    # Using reparameterisation:
    else:
        lp_del_1 = jax.jacfwd(lp, argnums=0)
        lp_del_1 = vectorise_functions(lp_del_1)
        return lp_del_1

def vectorise_functions(*func_list):
    vmap_list = []
    for fun in func_list:
        vmap_fun = jax.vmap(fun, in_axes(0, None))
        vmap_list.append(vmap_fun)
    return vmap_list