import numpy as np
from numpy.random import multivariate_normal as mvn_sample
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

COV_MAX = 1e2
COV_MIN = -1*COV_MAX

def create_gaussian(ndim, mean_lb, mean_ub, var_ub, cov_lb, cov_ub):

    def sample(num_samples, phi):
        cov = covariance_from_cholesky(phi['chol_diag'], phi['chol_lowerdiag'])
        samples = mvn_sample(phi["mean"], cov, size=num_samples)
        return samples.reshape(num_samples, ndim)
    
    mean = np.zeros(ndim)
    cov = np.identity(ndim)
    def sample_base(num_samples):
        samples = mvn_sample(mean, cov, size=num_samples)
        return samples.reshape(num_samples, ndim)

    def transform(epsilon, phi):
        L = assemble_cholesky(phi['chol_diag'], phi['chol_lowerdiag'])
        theta = phi["mean"] + jnp.einsum('ij,aj->ai', L, epsilon)
        return theta.reshape(-1, ndim)

    def lp(theta, phi):
        cov = covariance_from_cholesky(phi['chol_diag'], phi['chol_lowerdiag'])
        return mvn.logpdf(theta, mean=phi['mean'], cov=cov)

    lp_del_1 = jax.vmap(jax.jacfwd(lp, argnums=0), in_axes=(0,None))
    lp_del_2 = jax.vmap(jax.jacfwd(lp, argnums=1), in_axes=(0,None))
    transform_del_2 = jax.jacfwd(transform, argnums=1)

    # Create dictionary of functions:
    func_dict = {'lp': lp,
                 'sample': sample,
                 'sample_base': sample_base,
                 'transform': transform,
                 'lp_del_1': lp_del_1,
                 'lp_del_2': lp_del_2,
                 'transform_del_2': transform_del_2}

    # Create attribute dictionary:
    phi = {'mean': np.ones(ndim),
           'chol_diag': np.ones(ndim), 
           'chol_lowerdiag': np.zeros((ndim**2-ndim)//2)}
    phi_shape = {key: phi_i.shape for key, phi_i in phi.items()}
    phi_lb, phi_ub = create_cov_bounds(var_ub, ndim, cov_lb, cov_ub)
    phi_lb['mean'], phi_ub['mean'] = mean_lb, mean_ub
    attr_dict = {'phi': phi,
                 'phi_shape': phi_shape,
                 'phi_lb': phi_lb,
                 'phi_ub': phi_ub}

    return (func_dict, attr_dict)

def assemble_cholesky(chol_diag, chol_lowerdiag):
    
    # Check that inputs are compatable:
    check_cholesky_inputs(chol_diag, chol_lowerdiag)

    # Form covariance matrix from Cholesky decomposition:
    dim = len(chol_diag)
    L = jnp.diag(chol_diag)
    lower_diag_idx = jnp.tril_indices(dim, k=-1)
    L = jax.ops.index_update(L, lower_diag_idx, chol_lowerdiag)

    return L

def covariance_from_cholesky(chol_diag, chol_lowerdiag):
    L = assemble_cholesky(chol_diag, chol_lowerdiag)
    cov = L @ L.T
    return cov

def check_cholesky_inputs(diag, lower_diag):

    dim = len(diag)
    try:
        assert len(lower_diag) == int(0.5*(dim**2 - dim))
    except AssertionError:
        error_message = f'''Array of diagonal values contains {dim} elements, 
                            but array of lower diagonal elements contained 
                            {len(lower_diag)} elements. Instead, expected 
                            {0.5*(dim**2 - dim)} lower diagonal elements.'''
        raise ValueError(error_message)

def create_cov_bounds(max_var, ndim, cov_lb, cov_ub):

    # Compute length of both arrays:
    len_diag = ndim
    len_lowerdiag = int(0.5*(ndim**2 - ndim)) 

    # Create 'template' of ones for lower diagonal bounds:
    lowerdiag_ones = np.ones(len_lowerdiag)

    # Create lower-bounds:
    diag_lb = np.zeros(ndim)
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