import numpy as np
import jax
from numpy.random import multivariate_normal as mvn

from approx_post import ApproximateDistribution, JointDistribution, reverse_kl, forward_kl

def create_data(model, true_theta, noise_cov, num_samples):
    mean = model(true_theta)
    samples = mvn(mean, noise_cov, num_samples)
    return samples.reshape(num_samples, -1)

def main():

    # First, let's define a model:
    ndim = 3
    model = lambda theta: theta**2
    model_grad = jax.vmap(jax.jacfwd(model), in_axes=0)

    # Create artificial data:
    true_theta = np.random.rand(ndim)
    noise_cov = np.identity(ndim)
    num_samples = 3
    data = create_data(model, true_theta, noise_cov, num_samples)

    print(f'True theta: \n {true_theta}')
    print(f'Observations: \n {data}')

    # Create Gaussian approximate distribution:
    approx = ApproximateDistribution.gaussian(ndim)

    # Create Joint distribution from forward model:
    prior_mean = np.zeros(ndim)
    prior_cov = np.identity(ndim)
    joint = JointDistribution.from_model(data, model, noise_cov, prior_mean, prior_cov, model_grad)

    # Fit distribution to reverse KL divergence:
    results_dict = forward_kl.fit(approx, joint, use_reparameterisation=True, verbose=True)

if __name__ == "__main__":
    main()