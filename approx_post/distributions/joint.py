import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
import warnings
from .wrappers import wrap_dist_func, wrap_model_func

class JointDistribution:

    @classmethod
    def from_model(cls, model, noise_cov, prior_mean, prior_cov, model_grad=None):
        func_dict = joint_from_model(model, noise_cov, prior_mean, prior_cov, model_grad)
        theta_dim = len(prior_mean)
        return cls(func_dict, theta_dim)
        
    @classmethod
    def from_joint(cls, lp, theta_dim, lp_del_theta=None):
        func_dict = {'lp': lp, 'lp_del_1': lp_del_theta}
        return cls(func_dict, theta_dim)

    @classmethod
    def from_prior_and_likelihood(cls, prior_lp, likelihood_lp, theta_dim, prior_lp_grad=None, likelihood_lp_grad=None):
        func_dict = joint_from_prior_and_likelihood(prior_lp, likelihood_lp, prior_lp_grad, likelihood_grad)
        return cls(func_dict, theta_dim)

    def __init__(self, func_dict, theta_dim):
        self._func_dict = preprocess_func_dict(func_dict, theta_dim)

    def logpdf(self, theta, x):
        assert x.ndim == 3
        lp = self.fun_dict['lp'](theta, x)
        return lp

# Functions to create joint distributions from model function:
def joint_from_model(model, noise_cov, prior_mean, prior_cov, model_del_theta=None):

    # Ensure model functions output correct shapes:
    theta_dim = len(prior_mean)
    x_dim = noise_cov.shape[-1]

    model, model_del_theta = \
    [wrap_model_func(func, func_name, x_dim, theta_dim) for func, func_name in {'model': model, 'model_del_theta': model_del_theta}.items()]

    mvn_vmap_1 = jax.vmap(mvn.logpdf, in_axes=(1,None,None), out_axes=1)
    mvn_vmap_2 = jax.vmap(mvn.logpdf, in_axes=(None,1,None), out_axes=1)
    def lp(theta, x_obs): 
        # theta.shape = (num_batch, num_samples, theta_dim), 
        # x_obs.shape = (num_batch, num_obs, x_dim)
        num_batch, num_samples = x_obs.shape[0], theta.shape[0]
        prior_lp = mvn_vmap_1(theta, prior_mean, prior_cov) # prior_lp.shape = (num_batch, num_samples)
        x_pred = model(theta) # x_pred.shape = (num_batch, num_samples, x_dim)
        like_lp = mvn_vmap_2(x_obs, x_pred, noise_cov) # like_lp.shape = (num_batch, num_samples, num_obs)
        # Sum over num_obs axis:
        joint_lp = prior_lp + jnp.sum(like_lp, axis=-1) # joint_lp.shape = (num_batch, num_samples)
        return joint_lp.reshape(num_batch, num_samples)
    
    # If model gradients specified, can construct functions for gradient of joint lp wrt theta,
    # which can be used to perform reparameterisation trick:
    if model_del_theta is not None:

        mvn_del_mean = jax.jacfwd(mvn.logpdf, argnums=1) 
        mvn_del_mean_vmap = jax.vmap(jax.vmap(mvn_del_mean, in_axes=(None,0,None)), in_axes=(0,0,None))

        def prior_del_theta(theta):
            lp_grad = mvn_del_mean(theta, prior_mean, prior_cov)
            return lp_grad 

        def likelihood_del_mean(x_obs, x_pred, noise_cov): 
            # x_obs.shape = (num_batch, num_obs, x_dim) 
            # x_pred.shape = (num_batch, num_sample, x_dim)
            lp_grad = mvn_del_mean_vmap(x_obs, x_pred, noise_cov) # lp_grad.shape = (num_batch, num_samples, num_obs, x_dim)
            # Sum over lp of all observations made:
            lp_grad = jnp.sum(lp_grad, axis=2) # lp_grad.shape = (num_batch, num_samples, x_dim)
            return lp_grad

        def lp_del_theta(theta, x_obs):
            # theta.shape = (num_batch, num_samples, theta_dim), 
            # x_obs.shape = (num_batch, num_obs, x_dim)
            prior_grad = prior_del_theta(theta) # prior_grad.shape = (num_batch, num_samples, theta_dim)
            x_pred = model(theta) # x_pred.shape = (num_batch, num_samples, x_dim)
            like_del_mean = likelihood_del_mean(x_obs, x_pred, noise_cov) # like_del_mean.shape = (num_batch, num_samples, x_dim)
            mean_del_theta = model_del_theta(theta) # mean_del_theta.shape = (num_batch, num_samples, x_dim, theta_dim)
            like_grad = jnp.einsum('abji,abj->abi', mean_del_theta, like_del_mean) # like_grad.shape = (num_batch, num_samples, theta_dim)
            lp_del_theta = prior_grad + like_grad # lp_del_theta.shape = (num_batch, num_samples, theta_dim)
            return lp_del_theta
    else:
        lp_del_theta = None

    func_dict = {'lp': lp,
                 'lp_del_1': lp_del_theta}

    return func_dict

# Functions to create joint distributions by specfying a prior and likelihood:
def joint_from_prior_and_likelihood(prior_lp_fun, likelihood_lp_fun, theta_dim, prior_del_theta_fun=None, likelihood_del_theta_fun=None):

    # Reshape functions:
    prior_lp = lambda theta : prior_lp_fun(theta).reshape(-1,)

    def likelihood_lp(theta, x):
        # theta.shape = (num_samples, theta_dim), 
        # x_obs.shape = (num_batch, num_obs, x_dim)
        num_batch, num_samples = x_obs.shape[0], theta.shape[0]
        return likelihood_lp_fun(theta, x).reshape(num_batch, num_samples)
    
    # Log-prob function which calls above two functions:
    def lp(theta, x_obs):
        return prior_lp(theta) + likelihood_lp(x_obs, theta)
    
    # If given both gradient of prior and likelihood, can construct gradient of lp
    # for use in reparameterisation schemes:
    if None not in (prior_lp_grad, likelihood_lp_grad):
        # Reshaped functions:
        prior_del_theta = lambda theta : prior_del_theta_fun.reshape(-1, theta_dim)
        def likelihood_del_theta(theta, x):
            num_batch, (num_samples, theta_dim) = x.shape[0], theta.shape
            return likelihood_lp_fun(theta, x).reshape(num_batch, num_samples, theta_dim)
        # Gradient function which utilises above two functions:
        def lp_del_theta(theta, x): 
            return prior_lp_grad(theta) + likelihood_lp_grad(theta, x)
    else:
        lp_del_theta = None

    func_dict = {'lp': lp,
                 'lp_del_1': lp_del_theta}

    return func_dict

# Convenience function to get num_samples and num_batch:
def get_batch_and_samples(x):
    return (x.shape[0], x.shape[1])

def preprocess_func_dict(func_dict):

    try:
        assert 'lp' in func_dict
    except AssertionError:
        error_msg = ("Joint distribution must include a log-probability function.")
        raise KeyError(error_msg)
    
    if 'lp_del_theta' not in func_dict:
        warn_msg = (f"Warning: Gradient of Joint distribution wrt theta not specified; reparameterisation trick cannot be used.")
        warnings.warn(" ".join(warn_msg))
    
    func_dict = {func_name: wrap_dist_func(func, func_name, theta_dim, dist_type='joint') for func_name, func in func_dict.items()}

    return func_dict