import os
import json
import math
import numbers
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn_logpdf
from jax.random import multivariate_normal as mvn_sample
from arraytainers import Jaxtainer

class ApproximateDistribution:

    #
    #   Constructors and Constructor Helpers
    #

    def __init__(self, phi, logpdf, sample=None, sample_base=None, transform=None):
        if (sample is None) and (all(None in [sample_base, transform])):
            raise ValueError('Must specify either a sample function OR both a sample_base function AND a transform function.')
        self._phi = Jaxtainer(phi)
        self._func_dict = self._place_funcs_in_dict(logpdf, sample, sample_base, transform)
        self._jaxfunc_dict = self._create_jax_functions(logpdf, sample, sample_base, transform)

    @staticmethod 
    def _place_funcs_in_dict(logpdf, sample, sample_base, transform):
        jaxfunc_dict = {'logpdf': logpdf, 'sample': sample}
        if sample_base is not None:
            jaxfunc_dict['sample_base'] = sample_base
        if transform is not None:
            jaxfunc_dict['transform'] = transform
        return jaxfunc_dict

    def _create_jax_functions(self, logpdf, sample, sample_base, transform):
        jaxfunc_dict = self._place_funcs_in_dict(logpdf, sample, sample_base, transform)
        jaxfunc_dict = self._differentiate_jaxfuncs(jaxfunc_dict)
        jaxfunc_dict = self._vectorise_jaxfuncs(jaxfunc_dict)
        return jaxfunc_dict

    @staticmethod
    def _differentiate_jaxfuncs(jaxfunc_dict):
        grad_funcs = {}
        for key, func in jaxfunc_dict.items():
            if key == 'logpdf':
                grad_funcs['logpdf_del_1'] = jax.jacfwd(func, argnums=0)
                grad_funcs['logpdf_del_2'] = jax.jacfwd(func, argnums=1)
            elif key == 'transform':
                grad_funcs['transform_del_2'] = jax.jacfwd(func, argnums=1)
        return {**jaxfunc_dict, **grad_funcs}

    @staticmethod
    def _vectorise_jaxfuncs(jaxfunc_dict):
        for key, func in jaxfunc_dict.items():
            # Vectorise 'logpdf' funcs over batch dimensions of theta and phi, then over sample dimension of theta:
            if 'logpdf' in key:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None)), in_axes=(0,0))
            # Vectorise 'transform' over batch dimension of phi, then over sample dimension of epsilon:
            elif 'transform' in key:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None)), in_axes=(None,0))
            # Vectorise 'sample' over batch dimensions of phi:
            elif key == 'sample':
                jaxfunc_dict[key] = jax.vmap(func, in_axes=(None,0,None))
        return jaxfunc_dict

    #
    #   Pre-Processing Methods
    #

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

    #
    #   Logpdf Methods
    #

    def logpdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        leading_dims = theta.ndim - 1
        theta = self._reshape_input(theta, ndim=3)
        lp = self._jaxfunc_dict['logpdf'](theta, phi)
        num_batch, num_sample = theta.shape[0:2]
        output_shape = (num_batch, num_sample)
        return lp.reshape(output_shape[-leading_dims:])

    def logpdf_del_1(self, theta, phi=None):
        phi = self._get_phi(phi)
        leading_dims = theta.ndim - 1
        theta = self._reshape_input(theta, ndim=3)
        logpdf_del_1 = self._jaxfunc_dict['logpdf_del_1'](theta, phi)
        num_batch, num_sample, dim_theta = theta.shape
        output_shape = (num_batch, num_sample)
        return logpdf_del_1.reshape(*output_shape[-leading_dims:], dim_theta)

    def logpdf_del_2(self, theta, phi=None):
        phi = self._get_phi(phi)
        leading_dims = theta.ndim - 1
        theta = self._reshape_input(theta, ndim=3)
        logpdf_del_2 = self._jaxfunc_dict['logpdf_del_2'](theta, phi)
        num_batch, num_sample = theta.shape[0:2]
        initial_dims = (num_batch, num_sample)
        return logpdf_del_2.reshape(*initial_dims[-leading_dims:], phi.shape[1:])

    #
    #   Sampling Methods
    #

    def sample(self, num_samples, prngkey, phi=None):
        phi = self._get_phi(phi)
        if 'sample' in self._jaxfunc_dict:
            theta = self._jaxfunc_dict['sample'](num_samples, phi, prngkey)
        elif ('sample_base' in self._jaxfunc_dict) and ('transform' in self._jaxfunc_dict):
            epsilon = self._jaxfunc_dict['sample_base'](num_samples, prngkey)
            theta = self._jaxfunc_dict['transform'](epsilon, phi)
        return theta

    def sample_base(self, num_samples, prngkey):
        epsilon = self._jaxfunc_dict['sample_base'](num_samples, prngkey)
        return epsilon.reshape(num_samples, -1)

    #
    #   Transform Methods
    #

    def transform(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        epsilon = self._reshape_input(epsilon, ndim=2)
        theta = self._jaxfunc_dict['transform'](epsilon, phi) 
        num_samples, dim_theta = epsilon.shape
        return theta.reshape(-1, num_samples, dim_theta)
    
    def transform_del_2(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        epsilon = self._reshape_input(epsilon, ndim=2)
        theta_del_phi = self._jaxfunc_dict['transform_del_2'](epsilon, phi) 
        num_samples, dim_theta = epsilon.shape
        return theta_del_phi.reshape(-1, num_samples, dim_theta, phi.shape[1:])

    #
    #   Phi Methods
    #

    @property
    def params(self):
        return self._phi

    def phi(self, x=None, d=None):
        phi = self._phi[None,:]
        # Match batch dimension of x if provided:
        if x is not None:
            phi = np.repeat(phi, repeats=x.shape[0], axis=0)
        return phi

    def update(self, new_phi):
        self._phi = Jaxtainer(new_phi)

    def get_function(self, name):
        try:
            func = self._func_dict[name]
        except KeyError:
            raise KeyError(f"No such function called '{name}'; " 
                           f"valid function names are: {' ,'.join(list(self._func_dict.keys()))}")
        return func

    #
    #   Save and Load Methods
    #

    def load(self, load_dir):
        self._phi = Jaxtainer(self._load_json(load_dir))

    @staticmethod
    def _load_json(load_dir):
        try:
            with open(load_dir, 'r') as f:
                dist_json = json.load(f)
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

    #
    #   Constructors and Constructor Helpers
    #

    def __init__(self, ndim, phi=None):
        if phi is None:
            phi = self._create_default_phi(ndim)
        assemble_cholesky, assemble_covariance, logpdf, sample, sample_base, transform = self._create_gaussian_funcs(ndim)
        super().__init__(phi, logpdf, sample, sample_base, transform)
        self._func_dict['cholesky'] = assemble_cholesky
        self._func_dict['cov'] = assemble_covariance

    @staticmethod
    def _create_default_phi(ndim):
        default_phi = {'mean': jnp.zeros(ndim), 
                       'log_chol_diag': jnp.zeros(ndim)}
        if ndim > 1:
            default_phi['chol_lowerdiag'] = jnp.zeros(ndim*(ndim-1)//2)
        return default_phi

    @staticmethod
    def _create_gaussian_funcs(ndim):

        def assemble_cholesky(phi): 
            chol_diag = jnp.exp(phi['log_chol_diag'])
            L = jnp.atleast_2d(jnp.diag(chol_diag))
            if 'chol_lowerdiag' in phi:
                lower_diag_idx = jnp.tril_indices(ndim, k=-1) 
                L = jax.ops.index_update(L, lower_diag_idx, phi['chol_lowerdiag'])
            return L 

        def assemble_covariance(phi):
            L = assemble_cholesky(phi)
            cov = L @ L.T
            # Ensure covariance matrix is symmetric:
            return 0.5*(cov + cov.T)

        def logpdf(theta, phi):
            cov = assemble_covariance(phi)
            return  mvn_logpdf.logpdf(theta, mean=phi['mean'], cov=cov).squeeze()

        def transform(epsilon, phi):
            chol = assemble_cholesky(phi)
            return phi['mean'] + jnp.einsum('ij,j->i', chol, epsilon)

        transform_vmap = jax.vmap(transform, in_axes=(0,None))

        def sample(num_samples, phi, prngkey):
            epsilon = sample_base(num_samples, prngkey)
            return transform_vmap(epsilon, phi)

        def sample_base(num_samples, prngkey):
            return mvn_sample(key=prngkey, mean=jnp.zeros(ndim), cov=jnp.identity(ndim), shape=(num_samples,))

        return assemble_cholesky, assemble_covariance, logpdf, sample, sample_base, transform 

    #
    #   Normal Distribution Properties   
    #

    @property
    def mean(self):
        return self.phi()['mean']

    @property
    def cov(self):
        return self._func_dict['cov'](self.phi())