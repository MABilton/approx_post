import os
import more_itertools
import json
from arraytainers import Jaxtainer
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn_logpdf
from jax.random import multivariate_normal as mvn_sample

class ApproximateDistribution:

    _default_phi_bounds = {'lb': -5, 'ub': 5}

    def __init__(self, phi, logpdf, sample=None, sample_base=None, transform=None):
        if (sample is None) and (all(None in [sample_base, transform])):
            raise ValueError('Must specify either a sample function OR both a sample_base function AND a transform function.')
        self._phi = Jaxtainer(phi)
        self._func_dict = self._create_func_dict(logpdf, sample, sample_base, transform)

    @staticmethod
    def _create_func_dict(logpdf, sample, sample_base, transform):
    
        func_dict = {}

        # Vectorise over batch dimensions of theta and phi, then over sample dimension of theta:
        func_dict['logpdf'] = jax.vmap(jax.vmap(logpdf, in_axes=(0,None)), in_axes=(0,0))
        func_dict['logpdf_del_1'] = jax.vmap(jax.vmap(jax.jacfwd(logpdf, argnums=0), in_axes=(0,None)), in_axes=(0,0))
        func_dict['logpdf_del_2'] = jax.vmap(jax.vmap(jax.jacfwd(logpdf, argnums=1), in_axes=(0,None)), in_axes=(0,0))

        # Vectorise over batch dimensions of phi:
        if sample is not None:
            func_dict['sample'] = jax.vmap(sample, in_axes=(None,0,None))
        if sample_base is not None:
            func_dict['sample_base'] = sample_base

        # Vectorise over batch dimension of phi, then over sample dimension of epsilon:
        if transform is not None:
            func_dict['transform'] = jax.vmap(jax.vmap(transform, in_axes=(0,None)), in_axes=(None,0))
            func_dict['transform_del_2'] = jax.vmap(jax.vmap(jax.jacfwd(transform, argnums=1), in_axes=(0,None)), in_axes=(None,0))
        
        return func_dict

    def phi(self, x=None):
        phi = self._phi[None,:]
        # Match batch dimension of x if provided:
        if x is not None:
            phi = np.repeat(phi, repeats=x.shape[0], axis=0)
        return phi

    def _get_phi(self, phi):
        if phi is None:
            phi = self.phi()
        return phi

    @staticmethod
    def _reshape_input(val, ndim):
        if val is not None:
            val = jnp.atleast_1d(val)
            for _ in range(ndim - val.ndim):
                val = val[None,:]
        return val

    def logpdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        leading_dims = theta.ndim - 1
        theta = self._reshape_input(theta, ndim=3)
        lp = self._func_dict['logpdf'](theta, phi)
        num_batch, num_sample = theta.shape[0:2]
        output_shape = (num_batch, num_sample)
        return lp.reshape(output_shape[-leading_dims:])

    def logpdf_del_1(self, theta, phi=None):
        phi = self._get_phi(phi)
        leading_dims = theta.ndim - 1
        theta = self._reshape_input(theta, ndim=3)
        logpdf_del_1 = self._func_dict['logpdf_del_1'](theta, phi)
        num_batch, num_sample, dim_theta = theta.shape
        output_shape = (num_batch, num_sample)
        return logpdf_del_1.reshape(*output_shape[-leading_dims:], dim_theta)

    def logpdf_del_2(self, theta, phi=None):
        phi = self._get_phi(phi)
        leading_dims = theta.ndim - 1
        theta = self._reshape_input(theta, ndim=3)
        logpdf_del_2 = self._func_dict['logpdf_del_2'](theta, phi)
        num_batch, num_sample = theta.shape[0:2]
        initial_dims = (num_batch, num_sample)
        return logpdf_del_2.reshape(*initial_dims[-leading_dims:], phi.shape[1:])

    def sample(self, num_samples, prngkey, phi=None):
        phi = self._get_phi(phi)
        if 'sample' in self._func_dict:
            theta = self._func_dict['sample'](num_samples, phi, prngkey)
        elif ('sample_base' in self._func_dict) and ('transform' in self._func_dict):
            epsilon = self._func_dict['sample_base'](num_samples, prngkey)
            theta = self._func_dict['transform'](epsilon, phi)
        return theta

    def sample_base(self, num_samples, prngkey):
        epsilon = self._func_dict['sample_base'](num_samples, prngkey)
        return epsilon.reshape(num_samples, -1)
    
    def transform(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        epsilon = self._reshape_input(epsilon, ndim=2)
        theta = self._func_dict['transform'](epsilon, phi) 
        num_samples, dim_theta = epsilon.shape
        return theta.reshape(-1, num_samples, dim_theta)
    
    def transform_del_2(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        epsilon = self._reshape_input(epsilon, ndim=2)
        theta_del_phi = self._func_dict['transform_del_2'](epsilon, phi) 
        num_samples, dim_theta = epsilon.shape
        return theta_del_phi.reshape(-1, num_samples, dim_theta, phi.shape[1:])

    def load(self, load_dir):
        self._phi = Jaxtainer(self._load_json(load_dir))

    def update(self, new_phi):
        # clipped_phi = np.clip(new_phi, self.phi_bounds['lb'],  self.phi_bounds['ub'])
        # self._phi = Jaxtainer(clipped_phi)
        self._phi = Jaxtainer(new_phi)

    @staticmethod
    def _load_json(load_dir):
        try:
            with open(load_dir, 'r') as f:
                dist_json = json.read(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {file_dir} not found in directory {file_dir}.')
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f'Unable to decode contents of {file_name}. '
                                       f'Ensure that {file_name} is in a JSON-readable format.')
        return dist_json

    def save(self, save_name='phi.json', save_dir='.', indent=4):
        to_save = self._phi.unpack()
        if save_name[-4:] != 'json':
            save_name += ".json"
        with open(os.path.join(save_dir, save_name), 'w') as f:
            json.dump(to_save, f, indent=indent)

class Gaussian(ApproximateDistribution):

    def __init__(self, ndim, phi=None):
        gaussian_funcs = self._create_gaussian_funcs(ndim)
        if phi is None:
            phi = self._create_default_phi(ndim)
        super().__init__(phi, **gaussian_funcs)

    def _create_gaussian_funcs(self, ndim):

        def sample_base(num_samples, prngkey):
            return mvn_sample(key=prngkey, mean=jnp.zeros(ndim), cov=jnp.identity(ndim), shape=(num_samples,))

        def transform(epsilon, phi):
            chol = self._assemble_cholesky(phi)
            return phi['mean'] + jnp.einsum('ij,j->i', chol, epsilon)

        transform_vmap = jax.vmap(transform, in_axes=(0,None))

        def sample(num_samples, phi, prngkey):
            epsilon = sample_base(num_samples, prngkey)
            return transform_vmap(epsilon, phi)

        def logpdf(theta, phi):
            cov = self._assemble_covariance(phi)
            return  mvn_logpdf.logpdf(theta, mean=phi['mean'], cov=cov).squeeze()

        gaussian_funcs = {'logpdf': logpdf, 
                          'sample': sample, 
                          'sample_base': sample_base, 
                          'transform': transform}

        return gaussian_funcs

    def mean(self, x=None):
        return self.phi(x)['mean']

    def cov(self, x=None):
        return self._assemble_covariance(self.phi(x))

    def _assemble_covariance(self, phi):
        L = self._assemble_cholesky(phi)
        cov = L @ L.T
        # Ensure covariance matrix is symmetric:
        cov = 0.5*(cov + cov.T)
        return cov

    def _assemble_cholesky(self, phi): 
        chol_diag = jnp.exp(phi['log_chol_diag'])
        L = jnp.atleast_2d(jnp.diag(chol_diag))
        if 'chol_lowerdiag' in phi:
            lower_diag_idx = jnp.tril_indices(ndim, k=-1) 
            L = jax.ops.index_update(L, lower_diag_idx, phi['chol_lowerdiag'])
        return L 

    @staticmethod
    def _create_default_phi(ndim):
        default_phi = {'mean': jnp.zeros(ndim), 
                       'log_chol_diag': jnp.zeros(ndim)}
        if ndim > 1:
            default_phi['chol_lowerdiag'] = jnp.zeros(ndim*(ndim-1)//2)
        return default_phi