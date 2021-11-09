import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

from .func_utils import create_mvn_grads, reshape_joint_lp, reshape_model_func, reshape_prior_and_likelihood

class JointDistribution:

    @classmethod
    def from_model(cls, x, model, noise_cov, prior_mean, prior_cov, model_grad=None):
        func_dict = joint_from_model(model, noise_cov, prior_mean, prior_cov, model_grad)
        return cls(func_dict, x)
        
    @classmethod
    def from_joint(cls, lp, x, lp_del_theta=None):
        func_dict = {'lp': lp, 'lp_del_1': lp_del_theta}
        return cls(func_dict, x)

    @classmethod
    def from_prior_and_likelihood(cls, x, prior_lp, likelihood_lp, prior_lp_grad=None, likelihood_lp_grad=None):
        func_dict = joint_from_prior_and_likelihood(prior_lp, likelihood_lp, 
                                                     prior_lp_grad, likelihood_grad)
        return cls(func_dict, x)

    def __init__(self, func_dict, x):
        self._func_dict = func_dict
        self.x = x

    def logpdf(self, theta, x=None):
        x_obs = self.x if x is None else x
        lp = self.fun_dict['lp'](theta, x_obs)
        return lp

# Helper functions:
def joint_from_model(model, noise_cov, prior_mean, prior_cov, model_del_theta):

    # Ensure model functions output correct shapes:
    theta_dim = len(prior_mean)
    x_dim = noise_cov.shape[1]
    model, model_del_theta = reshape_model_func(theta_dim, x_dim, model, model_del_theta)

    mvn_vmap_1 = jax.vmap(mvn.logpdf, in_axes=(0,None,None))
    mvn_vmap_2 = jax.vmap(mvn.logpdf, in_axes=(None,0,None))

    # x_obs = num_obs * dim_x
    def lp(theta, x_obs):
        prior_lp = mvn_vmap_1(theta, prior_mean, prior_cov)
        x_pred = model(theta)
        like_lp = mvn_vmap_2(x_obs, x_pred, noise_cov)
        return prior_lp + jnp.sum(like_lp, axis=1)
    
    if model_del_theta is not None:
        prior_del_theta, likelihood_del_mean = create_mvn_grads(theta_dim, x_dim)
        def lp_del_theta(theta, x_obs):
            prior_grad = prior_del_theta(theta, prior_mean, prior_cov)
            like_del_mean = likelihood_del_mean(x_obs, model(theta), noise_cov)
            like_grad = jnp.einsum('aji,aj->ai', model_del_theta(theta), like_del_mean)
            return prior_grad + like_grad
    else:
        lp_del_theta = None

    lp, lp_del_theta = reshape_joint_lp(theta_dim, lp, lp_del_theta)

    func_dict = {'lp': lp,
                 'lp_del_1': lp_del_theta}

    return func_dict

def joint_from_prior_and_likelihood(prior_lp, likelihood_lp, theta_dim, prior_del_theta=None, likelihood_del_theta=None):

    prior_lp, likelihood_lp, prior_del_theta, likelihood_del_theta = \
        reshape_prior_and_likelihood(theta_dim, prior_lp, likelihood_lp, prior_del_theta, likelihood_del_theta)

    def lp(theta, x_obs):
        return prior_lp(theta) + likelihood_lp(x_obs, theta)
    
    if None not in (prior_lp_grad, likelihood_lp_grad):
        def lp_del_theta(theta, x):
            return prior_lp_grad(theta) + likelihood_lp_grad(theta, x)
    else:
        lp_del_theta = None

    lp, lp_del_theta = reshape_joint_lp(theta_dim, lp, lp_del_theta)

    func_dict = {'lp': lp,
                 'lp_del_1': lp_del_theta}

    return func_dict