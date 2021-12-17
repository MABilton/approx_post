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

    vectorised_shape = (num_batch, num_samples, -1)
    val_vec = vectorise_cv_input(val, vectorised_shape)
    cv_vec = vectorise_cv_input(cv, vectorised_shape)
    
    var = np.mean(np.einsum("abi,abj->abij", cv_vec, cv_vec), axis=1) # var.shape = (num_batches, num_samples, dim_cv, dim_cv)
    val_vec_delta = val_vec-np.mean(val_vec, axis=1, keepdims=True)
    cov = np.mean(np.einsum("abi,abj->abij", cv_vec, val_vec_delta), axis=1) # cov.shape = (num_batches, num_samples, dim_cv, dim_val)
    a = np.linalg.solve(var, cov) # a.shape = (num_batch, num_samples, dim_cv, dim_val)
    val_vec = np.mean(val_vec - np.einsum("aij,abi->abj", a, cv_vec), axis=1) # val_vec.shape = (num_batch, num_val)
    
    val = repack_cv_output(val_vec, val)

    return val

def vectorise_cv_input(val, vectorised_shape):
    flattened = val.flatten(order='F')
    flattened = flattened.reshape(vectorised_shape, order='F')
    return flattened # flattened.shape = (num_batch, num_samples, -1)

def repack_cv_output(val_vec, val):
    val_is_array = isinstance(val, (np.ndarray, jnp.DeviceArray))
    new_shape = val.shape[0:3:2]
    if val_is_array:
        val = val_vec.reshape(new_shape)
    else:
        val = Numpytainer.from_vector(val_vec, new_shape, order='F')
    return val