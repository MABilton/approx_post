import numpy as np
from numpy.random import multivariate_normal as mvn_sample
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

COV_MAX = 1e2
COV_MIN = -1*COV_MAX
VAR_MIN = 1e-2

def create_gaussian(ndim, mean_lb, mean_ub, var_ub, cov_lb, cov_ub):
    
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

    def covariance_from_cholesky(chol_diag, chol_lowerdiag):
        # chol_diag.shape = (ndim, )
        # chol_lowerdiag.shape = (0.5*(ndim**2 - ndim), )
        L = assemble_cholesky(chol_diag, chol_lowerdiag)
        cov = L @ L.T
        # Ensure covariance matrix is symmetric:
        cov = 0.5*(cov + cov.T)
        return cov

    # cov_vmap = jax.vmap(covariance_from_cholesky, in_axes=(0,0), out_axes=0)

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

    # Create attribute dictionary:

    phi = {'mean': 0.1*np.ones(ndim),
           'chol_diag': np.ones(ndim), 
           'chol_lowerdiag': np.zeros((ndim**2-ndim)//2)}
    phi_shape = {key: phi_i.shape for key, phi_i in phi.items()}
    phi_lb, phi_ub = create_cov_bounds(var_ub, ndim, cov_lb, cov_ub)
    phi_lb['mean'], phi_ub['mean'] = mean_lb, mean_ub
    attr_dict = {'params': phi,
                 'phi_shape': phi_shape,
                 'phi_lb': phi_lb,
                 'phi_ub': phi_ub}

    return (func_dict, attr_dict)

def check_cholesky_inputs(diag, lower_diag):

    dim = diag.shape[-1]
    try:
        lowerdiag_len =  lower_diag.shape[-1]
        req_len = int(0.5*(dim**2 - dim))
        assert lowerdiag_len == req_len
    except AssertionError:
        error_message = (f'Array of diagonal values contains {dim} elements,',
                         'but array of lower diagonal elements contained',
                         f'{lowerdiag_len} elements. Instead, expected',
                         f'{req_len} lower diagonal elements.')
        raise ValueError(' '.join(error_message))

def create_cov_bounds(max_var, ndim, cov_lb, cov_ub):

    # Compute length of both arrays:
    len_diag = ndim
    len_lowerdiag = int(0.5*(ndim**2 - ndim)) 

    # Create 'template' of ones for lower diagonal bounds:
    lowerdiag_ones = np.ones(len_lowerdiag)

    # Create lower-bounds:
    diag_lb = VAR_MIN*np.ones(ndim)
    lowerdiag_lb = cov_lb*lowerdiag_ones if cov_lb is not None else COV_MIN*lowerdiag_ones

    # Create upper-bounds:
    diag_ub = max_var*np.ones(ndim) if max_var is not None else COV_MAX*np.ones(ndim)
    lowerdiag_ub = cov_ub*lowerdiag_ones if cov_ub is not None else COV_MAX*lowerdiag_ones
    
    # Place bounds in dictionary:
    lb = {"chol_diag": diag_lb, 
          "chol_lowerdiag": lowerdiag_lb}
    ub = {"chol_diag": diag_ub, 
          "chol_lowerdiag": lowerdiag_ub}

    return (lb, ub)