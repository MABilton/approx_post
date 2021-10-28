import numpy as np
import jax
import jax.scipy.stats as jstats
from forward_kl import fit_forward
from reverse_kl import fit_reverse
from approx import create_normal_approx

def create_joint(prior_mean, prior_var, noise_std, use_reparameterisation, forward_mapping):

    # Log probability function for a Gaussian posterior:
    def lp(theta, x):
        prior_lp = jstats.norm.logpdf(theta, loc=prior_mu, scale=prior_var**(1/2))
        like_lp = jstats.norm.logpdf(x, loc=forward_mapping(theta), scale=noise_std)
        like_lp = np.sum(like_lp, axis=0)
        return prior_lp + like_lp

    lp_vmap = jax.vmap(joint_lp, in_axes=(0,None))

    if use_reparameterisation:
        joint_del_theta = jax.jacfwd(joint_lp, argnums=0)
        joint_del_theta = jax.vmap(joint_del_theta, in_axes=(0,None))
        output_dict = {"lp": lp_vmap, "lp_del_theta": joint_del_theta}
    else:
        output_dict = {"lp": lp_vmap}

    return output_dict

def create_data(epsilon, forward_mapping, noise_std, num_samples):
    theta = forward_mapping(epsilon)
    samples = np.random.normal(loc=theta, scale=noise_std, size=num_samples)
    return samples

def main():

    # Define forward mapping, which maps epsilon -> theta:
    forward_mapping = lambda epsilon: epsilon

    # Create artificial data:
    epsilon = 2
    noise_std = 0.2
    num_samples = 1
    data = create_data(epsilon, forward_mapping, noise_std, num_samples)

    # Define initial guess for phi and limits on values:
    mean_lim = np.array([1e3, 1e3])
    phi_bounds = {"mean": np.array(-1*mean_lim, mean_lim),
                  "cov": np.array([ , ])}


if __name__ == "__main__":
    main()