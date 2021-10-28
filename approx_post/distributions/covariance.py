import jax.numpy as jnp

def assemble_cholesky(chol_diag, chol_lowerdiag):
    
    # Check that inputs are compatable:
    check_cholesky_inputs(chol_diag, chol_lowerdiag)

    # Form covariance matrix from Cholesky decomposition:
    dim = len(diag)
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

def create_cov_bounds(max_var, ndim, cov_lb=None, cov_ub=None):

    # Compute length of both arrays:
    len_diag = ndim
    len_lowerdiag = int(0.5*(ndim**2 - ndim)) 

    # Create 'template' of ones for lower diagonal bounds:
    lowerdiag_ones = np.ones(len_lowerdiag)

    # Create lower-bounds:
    diag_lb = np.zeros(ndim)
    lowerdiag_lb = cov_lb*lowerdiag_ones if cov_lb is not None else PHI_MIN*lowerdiag_ones

    # Create upper-bounds:
    diag_ub = max_var*np.ones(ndim)
    lowerdiag_ub = cov_ub*lowerdiag_ones if cov_ub is not None else PHI_MAX*lowerdiag_ones
    
    # Place bounds in dictionary:
    lb = {"chol_diag": diag_lb, 
          "chol_lowerdiag": lowerdiag_lb}
    ub = {"chol_diag": diag_ub, 
          "chol_lowerdiag": lowerdiag_ub}

    return (lb, ub)