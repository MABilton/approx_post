
from jax.core import InconclusiveDimensionOperation
import jax.numpy as jnp
from arraytainers import Jaxtainer

reshape_error = (ValueError, InconclusiveDimensionOperation)

def wrap_model_func(func, func_name, x_dim, theta_dim):

    func_is_grad = func_name == 'model_grad'

    def wrapped_model(*args):
        output = func(*args)
        theta = args[0]
        num_batch, num_samples = theta.shape[0:2]
        outshape = (num_batch, num_samples, x_dim, theta_dim) if func_is_grad else (num_batch, num_samples, x_dim)
        try:
            output = output.reshape(outshape)
        except reshape_error:
            error_msg = (f"Unable to coerce output of {'model gradient' if func_is_grad else 'model'} function,",
                         f"which is of shape {output.shape}, into the shape {outshape}.")
            raise ValueError(' '.join(error_msg))
        return jnp.array(output)

    returned_func = wrapped_model if func is not None else None

    return returned_func

def wrap_dist_func(func, func_name, theta_dim, dist_type=None):
    
    dist_is_approx = 'approx' in dist_type.lower()
    dist_name = 'Approximating' if dist_is_approx else 'Joint'

    def wrapped_func(*args):

        output = func(*args)

        if dist_is_approx:
            output, outshape = process_approx_output(args, output, func_name, theta_dim)
        else:
            output, outshape = process_joint_output(args, output, func_name, theta_dim)
        
        try:
            output = output.reshape(outshape) 
        except reshape_error:
            error_msg = (f"Unable to coerce output of {func_name} function of the {dist_name} distribution,",
                         f"which is of shape {output.shape}, into the shape {outshape}.")
            raise ValueError(' '.join(error_msg))
        
        return output
    
    return wrapped_func

def process_approx_output(args, output, func_name, theta_dim):

    if func_name == 'lp':
        theta = args[0]
        num_batch, num_sample = theta.shape[0:2]
        outshape = (num_batch, num_sample)
    elif func_name == 'transform':
        epsilon, phi = args
        num_batch, num_samples = get_numbatch_from_phi(phi), epsilon.shape[0]
        outshape = (num_batch, num_sample, theta_dim)
    elif func_name == 'transform_del_2':
        epsilon, phi = args
        num_batch, num_samples = get_numbatch_from_phi(phi), epsilon.shape[0]
        outshape = (num_batch, num_sample, theta_dim, phi.shape[0:2])
    elif func_name == 'sample':
        num_samples, phi = args
        num_batch = get_numbatch_from_phi(phi)
        outshape = (num_batch, num_samples, theta_dim)
    elif func_name == 'sample_base':
        num_samples  = args[0]
        outshape = (num_samples, theta_dim)
    elif func_name == 'lp_del_1':
        theta, phi = args
        num_batch, num_samples = theta.shape[0:2]
        outshape = (num_batch, num_samples, dim_theta)
    elif func_name == 'lp_del_2':
        theta, phi = args
        num_batch, num_samples = theta.shape[0:2]
        outshape = (num_batch, num_samples, phi.shape[2:])
    elif func_name == 'phi':
        params, x = args
        num_batch = x.shape[0]
        outshape = (num_batch, params.shape)

    output = Jaxtainer(output) if func_name in ('lp_del_2', 'transform_del_2', 'phi') else jnp.array(output)

    return output, outshape

def process_joint_output(args, func_name, theta_dim):

    if func_name in ('lp', 'likelihood_lp'):
        theta, phi = args
        num_batch, num_samples = theta.shape[0:2]
        outshape = (num_batch, num_samples)
    
    elif func_name == 'prior_lp':
        theta = args[0]
        num_batch, num_samples = theta.shape[0:2]
        outshape = (num_batch, num_samples)
    
    elif func_name == 'lp_del_theta':
        theta, phi = args
        num_batch, num_samples = theta.shape[0:2]
        outshape = (num_batch, num_samples, dim_theta)

    return outshape

def get_numbatch_from_phi(phi):
    return phi.array_list[0].shape[0]