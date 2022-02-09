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

    def __init__(self, phi, logpdf, sample=None, sample_base=None, transform=None, phi_bounds=None):
        
        if (sample is None) and (all(None in [sample_base, transform])):
            raise ValueError('Must specify either a sample function OR both a sample_base function AND a transform function.')

        self._phi = Jaxtainer(phi)
        self._func_dict = self._create_func_dict(logpdf, sample, sample_base, transform)
        self.phi_bounds = \
            Jaxtainer({key: self._create_bounds(phi_bounds[key], self._default_phi_bounds[key]) for key in ['lb', 'ub']})
        self._ensure_phi_inside_bounds()

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

    def _create_bounds(self, bounds, default_val):

        # Need np.ones_like rather than jnp.ones_like since _phi is a Jaxtainer:
        ones_arraytainer = np.ones_like(self._phi)

        if bounds is None:
            bounds = default_val*ones_arraytainer
        else:
            bounds = self._remove_none_values_from_bounds(bounds)
            bounds = Jaxtainer(bounds)
            param_keys, bounds_keys = set(self._phi.keys()), set(bounds.keys())
            try:
                assert bounds_keys.issubset(param_keys)
            except AssertionError:
                warn_msg = (f"Specified bounds contains the keys ({' '.join(bounds_keys-param_keys)}), which",
                            "aren't included in the ApproximateDistribution parameters. These keys will",
                            "be ignored.")
                warnings.warn(" ".join(warn_msg))
                bounds = bounds.filter([key for key in param_keys.intersection(bounds_keys)])
            for key in (param_keys - bounds_keys):
                bounds[key] = jnp.array([default_val])
            # Broadcasting will 'fill out' rest of bounds container:
            bounds = bounds*ones_arraytainer
        return bounds

    @staticmethod
    def _remove_none_values_from_bounds(bounds):
        
        bounds_iter = bounds.items() if isinstance(bounds, dict) else enumerate(bounds)
        nonnone_bounds = {}
        for key, val in bounds_iter:
            if val is not None:
                nonnone_bounds[key] = val
            elif isinstance(val, (list, dict)):
                nonnone_bounds[key] = ApproximateDistribution._remove_none_values_from_bounds(val)

        if isinstance(bounds, list):
            nonnone_bounds = list(nonnone_bounds.values())

        return nonnone_bounds

    def _ensure_phi_inside_bounds(self):
        phi_outside_bounds = (self.phi() < self.phi_bounds['lb']) | (self.phi() > self.phi_bounds['ub'])
        if phi_outside_bounds.any():
            raise ValueError('At least one specified phi value lies outside of the valid range of phi values.')

    def initialise_phi(self, prngkey, mask=None):
        rand_val = jax.random.uniform(prngkey)
        new_phi = (self.phi_ub - self.phi_lb)*rand_val + self.phi_lb
        if mask is not None:
            self._phi[mask] = new_phi[mask]
        else:
            self._phi = new_phi

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
        clipped_phi = self._clip_phi_to_bounds(new_phi)
        self._phi = Jaxtainer(clipped_phi)

    def _clip_phi_to_bounds(self, new_phi):
        new_phi[new_phi < self.phi_bounds['lb']] = self.phi_bounds['lb'][new_phi < self.phi_bounds['lb']]
        new_phi[new_phi > self.phi_bounds['ub']] = self.phi_bounds['ub'][new_phi > self.phi_bounds['ub']]
        return new_phi

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

    _min_var = 1e-3

    def __init__(self, ndim, phi=None, mean_bounds=(None,None), var_bounds=(None,None), cov_bounds=(None,None)):
        gaussian_funcs = self._create_gaussian_funcs(ndim, mean_bounds, var_bounds, cov_bounds)
        phi_bounds = self._create_phi_bounds(ndim, mean_bounds, var_bounds, cov_bounds)
        if phi is None:
            phi = self._create_default_phi(ndim)
        super().__init__(phi, phi_bounds=phi_bounds, **gaussian_funcs)

    @staticmethod
    def _create_gaussian_funcs(ndim, mean_bounds, var_bounds, cov_bounds):

        def assemble_cholesky(phi): 
            L = jnp.atleast_2d(jnp.diag(phi['chol_diag']))
            if 'chol_lowerdiag' in phi:
                lower_diag_idx = jnp.tril_indices(ndim, k=-1) 
                L = jax.ops.index_update(L, lower_diag_idx, phi['chol_lowerdiag'])
            return L 

        def assemble_covariance(phi):
            L = assemble_cholesky(phi)
            cov = L @ L.T
            # Ensure covariance matrix is symmetric:
            cov = 0.5*(cov + cov.T)
            return cov

        def sample_base(num_samples, prngkey):
            return mvn_sample(key=prngkey, mean=jnp.zeros(ndim), cov=jnp.identity(ndim), shape=(num_samples,))

        def transform(epsilon, phi):
            chol = assemble_cholesky(phi)
            return phi['mean'] + jnp.einsum('ij,j->i', chol, epsilon)

        transform_vmap = jax.vmap(transform, in_axes=(0,None))

        def sample(num_samples, phi, prngkey):
            epsilon = sample_base(num_samples, prngkey)
            return transform_vmap(epsilon, phi)

        def logpdf(theta, phi):
            cov = assemble_covariance(phi)
            return  mvn_logpdf.logpdf(theta, mean=phi['mean'], cov=cov).squeeze()

        gaussian_funcs = {'logpdf': logpdf, 
                          'sample': sample, 
                          'sample_base': sample_base, 
                          'transform': transform}

        return gaussian_funcs

    def _create_phi_bounds(self, ndim, mean_bounds, var_bounds, cov_bounds):

        if (var_bounds[0] is None):
            var_bounds = (self._min_var, var_bounds[1])

        # Var bounds may be array, scalar, or Arraytainer - any won't work on scalars since they're non-iterable:
        if any(more_itertools.always_iterable(var_bounds[0] < 0)):
            warnings.warn('Negative variance lower bound specified, which is non-meaningful;'
                          f'variance lower bound changed to {self._min_var}')
            var_bounds = (self._min_var, var_bounds[1])

        mean_bounds = [val*jnp.ones(ndim) if val is not None else None for val in mean_bounds]
        choldiag_bounds = [val**0.5 if val is not None else None for val in var_bounds]
        phi_bounds = {'lb': {'mean': mean_bounds[0], 'chol_diag': choldiag_bounds[0]},
                      'ub': {'mean': mean_bounds[1], 'chol_diag': choldiag_bounds[1]}}
        
        if ndim > 1:
            chollowerdiag_bounds = [jnp.sign(val)*jnp.abs(val)**0.5 if val is not None else None for val in cov_bounds]
            phi_bounds['lb']['chol_lowerdiag'] = chollowerdiag_bounds[0]
            phi_bounds['ub']['chol_lowerdiag'] = chollowerdiag_bounds[1]

        return phi_bounds

    @staticmethod
    def _create_default_phi(ndim):
        default_phi = {'mean': jnp.zeros(ndim), 
                       'chol_diag': jnp.ones(ndim)}
        if ndim > 1:
            default_phi['chol_lowerdiag'] = jnp.zeros(ndim*(ndim-1)//2)
        return default_phi