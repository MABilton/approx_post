
# External imports:
from numpy.random import multivariate_normal as mvn
import jax.scipy.stats as jstats

# Local imports:
from .jax_utils import create_grads, vectorise_functions

def create_gaussian_approx(ndim, use_reparameterisation):

    # Sampling functions:
    def sample(num_samples, phi):
        samples = mvn(loc=phi["mean"], scale=phi["cov"], size=num_samples)
        return samples.reshape(num_samples, ndim)
    
    def sample_base(num_samples):
        mean = jnp.zeros(ndim)
        cov = jnp.identity(ndim)
        samples = mvn(loc=mean, scale=cov, size=num_samples)
        return samples.reshape(num_samples, ndim)

    # Transformation function:
    def transform(epsilon, phi):
        L = assemble_cholesky(phi['chol_diag'], phi['chol_lowerdiag'])
        theta = phi["mean"] + L @ epsilon
        return theta.reshape(ndim,)

    # Log probability function:
    def lp(theta, phi):
        return jstats.norm.logpdf(theta, loc=phi[0], scale=phi[1])

    if use_reparameterisation:

        approxfun_dict = 
    else:
        lp_vmap = vectorise_functions(lp)
        lp_del_phi = create_grads(lp) 
        approxfun_dict = {'sample': sample,
                          'lp': lp,
                          'lp_del_phi': lp_del_phi}

    # Add phi parameter shapes:
    phi_shape = {"mean": (ndim,), 
                 "chol_diag": (ndim,),
                 "chol_lowerdiag": (int(0.5*(ndim**2-ndim)),)}
    approxfun_dict["phi_shape"] = phi_shape

    return approxfun_dict