import numpy as np
import jax.numpy as jnp
from functools import reduce
from more_itertools.more import always_iterable
from arraytainers import Numpytainer

from ..distributions.amortised import AmortisedApproximation

def compute_loss_del_w(approx, params, x, loss_del_phi): # loss_del_phi.shape = (num_batch, *phi_shape)
    
    if not isinstance(approx, AmortisedApproximation):
        loss_del_w = loss_del_phi
    
    else:
        phi_del_w = approx._func_dict['phi_del_w'](params, x) # phi_del_w.shape = (*phi_shapes, *w_shapes)
        phi_del_w = Numpytainer(phi_del_w) 
        
        # Subtract 1 due to initial batch dimension:
        phi_ndims = np.ndims(loss_del_phi) - 1
        # Add singleton dimension for broadcasting puposes:
        phi_del_w = phi_del_w[None,:] # phi_del_w.shape = (1, *phi_shapes, *w_shapes)
        
        # Perform multiplcation on transposes for broadcasting purposes - allows us to broadcast over w dimensions:
        loss_del_phi = loss_del_phi.T # loss_del_phi.shape = (*reverse(phi_shapes), num_batch)
        phi_del_w = phi_del_w.T # phi_del_w.shape = (*reverse(w_shapes), *reverse(phi_shapes), 1)
        loss_del_w = phi_del_w * loss_del_phi # loss_del_w.shape = (*reverse(w_shapes), *reverse(phi_shapes), num_batch)
        loss_del_w = loss_del_w.T # loss_del_w.shape = (num_batches, *phi_shapes, *w_shapes)
        
        # Sum over phi values stored in different arraytainer elements:
        loss_del_w = loss_del_w.sum() # loss_del_w.shape = (num_batches, *phi_shapes, *w_shapes)
        # Sum over phi values stored in same arraytainer elements:
        loss_del_w = \
            loss_del_w.apply(lambda x, ndims: np.sum(x, axis=range(1,ndims+1)), args=(phi_ndims,)) 
            # loss_del_w.shape = (num_batches, *w_shapes)
    return loss_del_w 

def apply_cv(val, cv, num_batch, num_samples):

    val_is_array = isinstance(val, (np.ndarray, jnp.DeviceArray)) 

    val_vec = vectorise_cv_input(val, num_batch, num_samples, val_is_array)
    cv_vec = vectorise_cv_input(cv, num_batch, num_samples, val_is_array)
    var = np.mean(np.einsum("abi,abj->abij", cv_vec, cv_vec), axis=1) # var.shape = (num_batches, num_samples, dim_cv, dim_cv)
    val_vec_delta = val_vec-np.mean(val_vec, axis=1, keepdims=True)
    cov = np.mean(np.einsum("abi,abj->abij", cv_vec, val_vec_delta), axis=1) # cov.shape = (num_batches, num_samples, dim_cv, dim_val)
    a = np.linalg.solve(var, cov) # a.shape = (num_batch, num_samples, dim_cv, dim_val)
    val_vec = np.mean(val_vec - np.einsum("aij,abi->abj", a, cv_vec), axis=1) # val_vec.shape = (num_batch, num_val)
    
    val = val_vec.reshape([x for idx, x in enumerate(val.shape) if idx!=1]) if val_is_array else repack_arraytainer(val_vec, val.shape)

    return val

def vectorise_cv_input(val, num_batch, num_samples, val_is_array):
    val = Numpytainer(val) if val_is_array else val
    flattened_arraytainer = val.reshape(num_batch, num_samples,-1)
    arrays_to_cat = [x for x in flattened_arraytainer.list_arrays() if x.shape[-1]>0]
    flattened = np.concatenate(arrays_to_cat, axis=2)
    return flattened # flattened.shape = (num_batch, num_samples, -1)

def repack_arraytainer(vectorised_vals, shapes):
    vals = {}
    idx = 0
    for key, shape in shapes.items():
        out_shape = (shape[0], shape[-1])
        num_elem = np.prod(out_shape).item()
        vals[key] = vectorised_vals[:,idx:idx+num_elem].reshape(out_shape)
        idx += num_elem
    if shapes._type is list:
        vals = list(vals.values())
    return Numpytainer(vals)

# def vectorise_values(val):
#     shapes = val.shape
#     shapes = NumpyContainer(shapes)
#     num_samples = shapes[0][0]
#     val_vec = np.reshape(val, (num_samples,-1))
#     val_vec = np.concatenate(val_vec.values(), axis=1)
#     return val_vec

# def reshape_output(val_vec, val):
#     out_shapes = val.shape[1:]
#     if issubclass(type(val), ArrayContainer):
#         val_out = val.copy()
#         i = 0
#         for key, shape in out_shapes.items():
#             num_elem = reduce(lambda x,y: x*y, shape)
#             val_i = val_vec[i:i+num_elem]
#             i += num_elem
#             val_out[key] = val_i.reshape(shape)
#     else:
#         val_out = val_vec.reshape(out_shapes)
#     return val_out