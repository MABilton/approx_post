import numpy as np
import warnings
from numpy.random import multivariate_normal as mvn_sample
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

MIN_VAR = 1e-3

def create_gaussian(ndim, mean_bounds, var_bounds, cov_bounds):

    lowerdiag_len = (ndim**2-ndim)//2

    func_dict = create_func_dict(ndim, lowerdiag_len)

    # Initialise parameters:
    phi = {'mean': np.zeros(ndim),
           'chol_diag': np.ones(ndim), 
           'chol_lowerdiag': np.zeros(lowerdiag_len)}
    
    phi_lb, phi_ub = create_bounds(mean_bounds, var_bounds, cov_bounds)

    return func_dict, phi, phi_lb, phi_ub

def create_func_dict(ndim, lowerdiag_len):

    base_mean = np.zeros(ndim)
    base_cov = np.identity(ndim)

    # Assembly functions:
    def assemble_cholesky(chol_diag, chol_lowerdiag): 
        # chol_diag.shape = (ndim, )
        # chol_lowerdiag.shape = (0.5*(ndim**2 - ndim), )
        # Check that inputs are compatable:
        check_cholesky_inputs(chol_diag, chol_lowerdiag)
        # Form covariance matrix from Cholesky decomposition:
        L = jnp.diag(chol_diag) # L.shape = (ndim, ndim)
        lower_diag_idx = jnp.tril_indices(ndim, k=-1) 
        L = jax.ops.index_update(L, lower_diag_idx, chol_lowerdiag)
        return L

    def check_cholesky_inputs(diag, lower_diag):
        try:
            assert lower_diag.shape[-1] == lowerdiag_len
        except AssertionError:
            error_message = (f'Array of diagonal values contains {ndim} elements,',
                            'but array of lower diagonal elements contained',
                            f'{lower_diag.shape[-1]} elements. Instead, expected',
                            f'{lowerdiag_len} lower diagonal elements.')
            raise ValueError(' '.join(error_message))

    def covariance_from_cholesky(chol_diag, chol_lowerdiag):
        # chol_diag.shape = (ndim, )
        # chol_lowerdiag.shape = (0.5*(ndim**2 - ndim), )
        L = assemble_cholesky(chol_diag, chol_lowerdiag)
        cov = L @ L.T
        # Ensure covariance matrix is symmetric:
        cov = 0.5*(cov + cov.T)
        return cov

    # Sampling functions:

    def sample_base(num_samples):
        np.random.seed(1)
        samples = mvn_sample(base_mean, base_cov, size=num_samples)
        return samples.reshape(num_samples, ndim)

    def transform(epsilon, phi): 
        # NB: we'll vectorise over batch dimension of phi input with vmap
        # epsilon.shape = (num_samples, ndim)
        chol = assemble_cholesky(phi['chol_diag'], phi['chol_lowerdiag']) # chol.shape = (ndim, ndim)
        mean = phi['mean'] # mean.shape = (ndim,)
        # NB: Need '...' since we must vectorise function over num_samples 
        # dimension of epsilon when computing derivative of transform function:
        theta = mean + jnp.einsum('ij,...j->...i', chol, epsilon) # theta.shape = (num_samples, ndim)
        return theta

    transform_vmap = jax.vmap(transform, in_axes=(None,0), out_axes=0)

    def sample(num_samples, phi):
        epsilon = sample_base(num_samples) # epsilon.shape = (num_samples, ndim)
        theta = transform_vmap(epsilon, phi) # theta.shape = (num_batch, num_samples, ndim)
        return theta

    # Log probability function:

    def lp(theta, phi):
        # NB: we'll vectorise over batch dimension of phi input with vmap 
        # theta.shape = (num_samples, ndim)
        mean = phi['mean'] # mean.shape = (ndim,)
        cov = covariance_from_cholesky(phi['chol_diag'], phi['chol_lowerdiag']) # cov.shape = (ndim, ndim)
        log_prob = mvn.logpdf(theta, mean=mean, cov=cov) # log_prob.shape = (num_samples,)
        return log_prob

    # Vectorise log-probability over batch and sample dimensions:
    lp_vmap = jax.vmap(lp, in_axes=(0,0))

    # Gradient functions - NB: must vectorise over num_samples axis of theta input when computing gradients so that we don't compute cross-derivatives between samples (e.g. we don't compute the gradient of the log-probability of the i'th sample wrt to j'th sample):
    transform_del_2 = jax.vmap(jax.vmap(jax.jacfwd(transform, argnums=1), in_axes=(0,None)), in_axes=(None,0))
    lp_del_1 = jax.vmap(jax.vmap(jax.jacfwd(lp, argnums=0), in_axes=(0,None)), in_axes=(0,0))
    lp_del_2 = jax.vmap(jax.vmap(jax.jacfwd(lp, argnums=1), in_axes=(0,None)), in_axes=(0,0))

    # Create dictionary of functions:
    func_dict = {'lp': lp_vmap,
                 'sample': sample,
                 'sample_base': sample_base,
                 'transform': transform_vmap,
                 'lp_del_1': lp_del_1,
                 'lp_del_2': lp_del_2,
                 'transform_del_2': transform_del_2}
    
    return func_dict

def create_bounds(mean_bounds, var_bounds, cov_bounds):

    if var_bounds[0] is None:
        var_bounds = (MIN_VAR, var_bounds[1])
    elif any(var_bounds[0] < 0):
        error_msg = f'Specified variance lower bound {var_bounds[0]} contains negative values, which is non-meaningful.'
        raise ValueError(error_msg)

    bounds = []
    for idx in range(2):
        bounds_dict = dict(zip(['mean', 'chol_diag', 'chol_lowerdiag'], [mean_bounds[idx], var_bounds[idx], cov_bounds[idx]]))
        bounds_dict = {key: np.sign(val)*np.abs(val)**0.5 if (key!='mean') & (val is not None) else val 
                       for key, val in bounds_dict.items()}
        bounds.append(bounds_dict)
    phi_lb, phi_ub = bounds

    return phi_lb, phi_ub