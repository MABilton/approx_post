import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

# General function:
def reshape_output(function, output_shape):
    def reshaped(*args, **kwargs):
        return function(*args, **kwargs).reshape(output_shape)
    return reshaped
        
# Used to create JointDistribution object:
def create_mvn_grads(theta_dim, x_dim):

    mvn_grad = jax.jacfwd(mvn.logpdf, argnums=1)

    def prior_del_theta(theta, prior_mean, prior_cov):
        output = mvn_grad(theta, prior_mean, prior_cov)
        return output.reshape(-1, theta_dim)

    mvn_grad_vmap = jax.vmap(mvn_grad, in_axes=(None,0,None), out_axes=0)
    def likelihood_del_mean(x_obs, x_pred, noise_cov):
        # Compute lp_grad of each x_obs for each x_pred;
        # output.shape = (num_samples, num_obs, x_dim)
        output = mvn_grad_vmap(x_obs, x_pred, noise_cov)
        # Sum over lp of all observations made:
        output = jnp.sum(output, axis=1)
        # output.shape = (num_samples, x_dim)
        return output.reshape(-1, x_dim)

    return (prior_del_theta, likelihood_del_mean)

def reshape_prior_and_likelihood(theta_dim, prior_lp, likelihood_lp, prior_del_theta, likelihood_del_theta):
    
    lp_shape = (-1,)
    func_list = []
    for lp in (prior_lp, likelihood_lp):
        func_list.append(reshape_output(lp, lp_shape))
    prior_lp, likelihood_lp = func_list
    
    lp_grad_shape = (-1, theta_dim)
    func_list = []
    for lp_grad in (prior_del_theta, likelihood_del_theta):
        grad_func = reshape_output(lp_grad, lp_grad_shape) if lp_grad is not None else None
        func_list.append(grad_func)
    prior_del_theta, likelihood_del_theta = func_list
    
    return (prior_lp, likelihood_lp, prior_del_theta, likelihood_del_theta)

def reshape_joint_lp(theta_dim, lp, lp_del_theta):
    lp_shape = (-1, )
    lp = reshape_output(lp, lp_shape)
    if lp_del_theta is not None:
        lp_del_theta_shape = (-1, theta_dim)
        lp_del_theta = reshape_output(lp_del_theta, lp_del_theta_shape)
    return (lp, lp_del_theta)
    
def reshape_model_func(theta_dim, x_dim, model, model_del_theta, return_jax=True):
    model_shape = (-1, x_dim)
    model = reshape_output(model, model_shape)
    if model_del_theta is not None:
        model_del_theta_shape = (-1, x_dim, theta_dim)
        model_del_theta = reshape_output(model_del_theta, model_del_theta_shape)
    return (model, model_del_theta)