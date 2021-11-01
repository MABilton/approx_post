from .distribution import Distribution
from .func_utils import vectorise_functions

from jax.scipy.stats import multivariate_normal as mvn

class JointDistribution(Distribution):

    _func_to_load = ('sample', 'lp', 'lp_del_1', 'prior_sample', 
                     'prior_lp', 'likelihood_sample', 'likelihood_lp')
    _attr_to_load = ('x',)

    @classmethod
    def from_model(model, noise_std, prior_mean, prior_cov, x, model_grad=None):
        lp, lp_del_theta = _joint_from_model(model, noise_std, prior_mean, prior_cov, x, model_grad)
        func_dict = {'lp': lp}
        if lp_del_theta is not None:
            func_dict['lp_del_1'] = lp_del_theta
        attr_dict = {'x': x}
        return cls(func_dict, attr_dict)
        
    @classmethod
    def from_joint(lp, x, lp_del_theta=None):
        func_dict = {'lp': lp}
        if lp_del_theta is not None:
            func_dict['lp_del_1'] = lp_del_theta
        attr_dict = {'x': x}
        return cls(func_dict, attr_dict)

    @classmethod
    def from_prior_and_likelihood(cls, x, prior_lp, likelihood_lp, prior_lp_grad=None, likelihood_lp_grad=None):
        
        # Create joint functions from prior and likelihood:
        lp, lp_del_theta = _joint_from_prior_and_likelihood(prior_lp, likelihood_lp, 
                                                            prior_lp_grad, likelihood_grad)

        # Place functions in dictionary and create clas:
        func_dict = {'lp': lp, 'prior_lp': prior_lp, 'likelihood_lp': likelihood_lp}
        if lp_del_theta is not None:
            func_dict['lp_del_1'] = lp_del_theta

        attr_dict = {'x': x}

        return cls(func_dict, attr_dict)

    def __init__(self, func_dict, attr_dict):
        super().__init__(func_dict, attr_dict)
        self.x = attr_dict['x']

    def log_prob(self, theta, x=None):
        x_obs = self.x if x is None
        lp = self.fun_dict['lp'](theta, x_obs)
        return lp

# Helper functions:
def _joint_from_model(model, noise_cov, prior_mean, prior_cov, x, model_grad):

    def lp(theta, x_obs):
        prior_lp = mvn.logpdf(theta, loc=prior_mean, scale=prior_cov)
        x_pred = model(theta)
        like_lp = mvn.logpdf(x_obs, loc=x_pred, scale=noise_cov)
        return prior_lp + jnp.sum(like_lp, axis=0)
    
    mvn_grad = jax.jacfwd(mvn.logpdf, argnums=1)
    lp_vmap, mvn_grad = vectorise_functions(lp, mvn_grad)

    def lp_del_theta(theta, x_obs):
        prior_grad = mvn_grad(theta, loc=prior_mean, scale=prior_cov)
        x_pred = model(theta)
        like_del_mean = mvn_grad(x_obs, loc=x_pred, scale=noise_cov)
        like_grad = jnp.einsum('ai,ai->a', model_grad(theta), like_del_mean)
        return prior_grad + jnp.sum(like_grad, axis=0)

    if model_grad is None:
        grad_fun = None
    else:
        grad_fun = lp_del_theta
    
    return (lp, grad_fun)

def _joint_from_prior_and_likelihood(prior_lp, prior_sample, likelihood_lp, likelihood_sample, 
                                     prior_lp_grad=None, likelihood_lp_grad=None):
    
    def lp(theta, x_obs):
        return prior_lp(theta) + likelihood_lp(x_obs, theta)
    
    def sample(num_samples, x_obs):
        theta_samples = prior_sample(num_samples)
        x_samples = likelihood_sample(num_samples, theta_samples)
        return x_samples
    
    def lp_del_theta(theta, x):
        return prior_lp_grad(theta) + likelihood_lp_grad(theta, x)

    if None in (prior_lp_grad, likelihood_lp_grad):
        grad_fun = None
    else:
        grad_fun = lp_del_theta

    return (lp, sample, grad_fun)   