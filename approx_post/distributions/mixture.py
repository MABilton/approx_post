import math
import numpy as np
import jax
import jax.numpy as jnp
from arraytainers import Jaxtainer

class MixtureApproximation:

    _coeff_key = 'log_unnorm_coeff'

    #
    #   Initialisation Helper Methods
    #

    def __init__(self):
        self._nonfrozen = None

    @staticmethod
    def _place_funcs_in_dict(coefficients, logpdf, logpdf_components, transform, sample_base, sample_components):
        return {'coefficients': coefficients,
                'logpdf': logpdf,
                'logpdf_components': logpdf_components,
                'transform': transform,
                'sample_base': sample_base,
                'sample_components': sample_components}

    @staticmethod
    def _differentiate_jaxfuncs(jaxfunc_dict):
        
        grad_funcs = {}

        for key, func in jaxfunc_dict.items():

            if key == 'coefficients':
                grad_funcs['coefficients_del_phi'] = jax.jacfwd(func, argnums=0)

            elif key == 'logpdf':
                grad_funcs['logpdf_del_1'] = jax.jacfwd(func, argnums=0)
                grad_funcs['logpdf_del_2'] = jax.jacfwd(func, argnums=1)
            
            elif key == 'logpdf_components':
                grad_funcs['logpdf_components_del_1'] = jax.jacfwd(func, argnums=0)
            
            elif key == 'transform':
                grad_funcs['transform_del_2'] = jax.jacfwd(func, argnums=1)

        return {**jaxfunc_dict, **grad_funcs}

    @staticmethod
    def _vectorise_jaxfuncs(jaxfunc_dict):
        for key, func in jaxfunc_dict.items():
            # Vectorise 'coefficients' and 'coefficients_del_phi' over batch dimension of phi input:
            if 'coefficients' in key:
                jaxfunc_dict[key] = jax.vmap(func, in_axes=0)
            # Vectorise 'logpdf', 'logpdf_del_1', 'logpdf_del_2', 'logpdf_components', and 'logpdf_components_del_1' over 
            # batch dimension of theta and phi inputs, as well as over sample dimension of theta input:
            elif 'logpdf' in key:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None)), in_axes=(0,0))
            # Vectorise 'transform' and 'transform_del_2' over batch dimension of phi, and over sample dimension of epsilon:
            elif 'transform' in key:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None)), in_axes=(None,0))
            # Vectorise over batch dimension of probs:
            elif key == 'sample_components':
                jaxfunc_dict[key] = jax.vmap(func, in_axes=(None,None,None,0))
        return jaxfunc_dict

    #
    #   Component Methods
    #

    @staticmethod
    def component_key(idx):
        return f'component_{idx}'

    @property
    def nonfrozen(self):
        return self.component_key(self._nonfrozen)

    @property
    def is_frozen(self):
        return self._nonfrozen is not None

    def freeze(self, nonfrozen_idx):
        if isinstance(nonfrozen_idx, int):
            self._nonfrozen = nonfrozen_idx

    def unfreeze(self):
        self._nonfrozen = None

    #
    #   Pre-Processing and Post-Processing Methods
    #

    def _preprocess_theta(self, theta, phi):

        # theta.ndim must be at least 3: 
        # (num_batch, num_samples, num_mixture, theta_dim) OR
        # (num_batch, num_samples, theta_dim)
        for _ in range(3-theta.ndim):
            theta = theta[None,:]

        phi_batch_dim = phi.list_elements()[0].shape[0]
        theta_batch_dim = theta.shape[0]
        if (theta_batch_dim == 1) and (phi_batch_dim > 1):
            theta = jnp.repeat(theta, repeats=phi_batch_dim, axis=0)
        elif np.all(theta_batch_dim != phi_batch_dim):
            raise ValueError('Batch dimension of theta (= {theta_batch_dim}) and phi (= {phi_batch_dim}) must match.')  
        
        return theta

    def _get_phi(self, phi):
        if phi is None:
            phi = self.phi()
        return phi

    def _remove_frozen_grad_components(self, grad):
        if not self.is_frozen:
            # Set all gradients to zero except nonfrozen component:
            frozen_grad = 0*grad
            frozen_grad[self.nonfrozen] = grad[self.nonfrozen]
            grad = frozen_grad
        return grad

    #
    #   Coefficient Methods
    #

    def coefficients(self, phi=None):
        phi = self._get_phi(phi)
        return self._jaxfunc_dict['coefficients'](phi)
    
    def coefficients_del_phi(self, phi=None):
        phi = self._get_phi(phi)
        coeffs_del_phi = self._jaxfunc_dict['coefficients_del_phi'](phi)
        coeffs_del_phi = self._remove_frozen_grad_components(coeffs_del_phi)
        return coeffs_del_phi

    #
    #   Log-Probability and Sampling Methods
    #

    def logpdf_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        logpdf_components = self._jaxfunc_dict['logpdf_components'](theta, phi)
        num_batch, num_samples = theta.shape[:2]
        return logpdf_components.reshape(num_batch, num_samples, self.num_components)

    def pdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        pdf = self._jaxfunc_dict['pdf'](theta, phi)
        num_batch, num_samples = theta.shape[:2]
        return pdf.reshape(num_batch, num_samples)

    def logpdf_del_2_eps(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        logpdf = self._jaxfunc_dict['logpdf_eps'](epsilon, phi)
        num_batch, num_samples = theta.shape[:2]
        return logpdf.reshape()

    def logpdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        logpdf = self._jaxfunc_dict['logpdf'](theta, phi)
        return logpdf
    
    def logpdf_del_1_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        logpdf_del_1_components = self._jaxfunc_dict['logpdf_del_1_components'](theta, phi)
        return logpdf_del_1_components

    def logpdf_del_1(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        logpdf_del_1 = self._jaxfunc_dict['logpdf_del_1'](theta, phi)
        return logpdf_del_1.reshape(*theta.shape[:-1], theta_dim)

    def logpdf_del_2(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        logpdf_del_2 = self._jaxfunc_dict['logpdf_del_2'](theta, phi)
        logpdf_del_2 = self._remove_frozen_grad_components(logpdf_del_2)
        num_batch, num_samples = theta.shape[0:2]
        return logpdf_del_2.reshape(num_batch, num_samples, phi.shape[1:])

    #
    #   Transform Methods
    #

    def transform(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        theta = self._jaxfunc_dict['transform'](epsilon, phi)
        num_samples, num_components, dim_theta = epsilon.shape
        return theta.reshape(-1, num_samples, num_components, dim_theta)
    
    def transform_del_2(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        theta_del_2 = self._jaxfunc_dict['transform_del_2'](epsilon, phi)
        theta_del_2 = self._remove_frozen_grad_components(theta_del_2)
        num_samples, num_components, dim_theta = epsilon.shape
        return theta_del_2.reshape(-1, num_samples, num_components, dim_theta, phi.shape[1:])

    #
    #   Sample Methods
    #

    def sample_components(self, num_samples, prngkey, phi=None):
        phi = self._get_phi(phi)
        probs = self.coefficients(phi)
        num_choices = self.num_components
        return self._jaxfunc_dict['sample_components'](prngkey, num_samples, num_choices, probs)

    def sample_base(self, num_samples, prngkey):
        epsilon = self._jaxfunc_dict['sample_base'](num_samples, prngkey)
        theta_dim = epsilon.shape[-1]
        return epsilon.reshape(num_samples, self.num_components, theta_dim)
    
    def sample(self, num_samples, prngkey, phi=None):
        
        phi = self._get_phi(phi)

        num_samples_from_each_dist = self.sample_components(num_samples, prngkey, phi) # shape = (num_batch, num_samples)

        samples = []
        for idx, (key, dist) in enumerate(self.components.items()):
            num_samples_i = jnp.sum(num_samples_from_each_dist == idx, axis=1) # shape = (num_batch,)
            # Can't vectorise over num_samples input - draw maximum number of samples requested across all batches
            # for each batch instead:
            max_num_samples_i = jnp.max(num_samples_i)
            samples_i = dist.sample(max_num_samples_i, prngkey, phi[key]) # shape = (num_batch, max_num_samples_i, theta_dim)
            # For phi values where we've drawn too many samples:
            samples_i = self._replace_excess_samples_with_nan(samples_i, num_samples_i)
            samples.append(samples_i)

        return self._combine_samples_from_each_dist(samples, num_samples)

    @staticmethod
    def _replace_excess_samples_with_nan(samples, num_samples_for_each_batch):
        # samples.shape = (num_batch, max_num_samples_from_component, dim_theta)
        # num_samples_for_each_batch.shape = (num_batch,)
        sample_idx = jnp.arange(samples.shape[1]) # shape = (max_num_samples_from_component, )
        sample_idx = jnp.broadcast_to(sample_idx, samples.shape[2:0:-1]).T # shape = (max_num_samples_from_component, dim_theta)
        # Add trailing singleton dimensions for broadcasting purposes:
        nan_idx = sample_idx >= num_samples_for_each_batch[:,None,None] # shape = (num_batch, max_num_samples_from_component, dim_theta)
        return samples.at[nan_idx].set(math.nan)

    @staticmethod
    def _combine_samples_from_each_dist(samples, num_samples):
        # Append samples taken from different components along samples axis:
        samples = jnp.concatenate(samples, axis=1) # shape = (num_batch, sum(max(num_samples_i)), theta_dim)
        not_nan_idx = jnp.logical_not(jnp.isnan(samples))
        samples_without_nan = samples.at[not_nan_idx].get()
        num_batch, theta_dim = samples.shape[0], samples.shape[-1]
        return samples_without_nan.reshape(num_batch, num_samples, theta_dim)

class Different(MixtureApproximation):

    def __init__(self, approx_list):
        self._components = self._create_jax_functions(approx_list)
        super().__init__()

    @classmethod
    def _create_jax_functions(cls, components):

        #
        #   Coefficient functions:
        #

        def coefficients(phi):
            unnorm_coeffs = jnp.exp(phi[cls._coeff_key])
            norm_coeffs = unnorm_coeffs/jnp.sum(unnorm_coeffs)
            return norm_coeffs.squeeze()

        #
        #   Probability density functions:
        #

        def preprocess_theta(theta):
            if theta.ndim < 2:
                theta = theta[None,:]
                theta = jnp.repeat(theta, repeats=num_components, axis=0)
            return theta

        def logpdf_components(theta, phi):
            preprocess_theta = preprocess_theta(theta)
            logpdf = []
            for idx, (key, dist) in enumerate(components.items()):
                logpdf_i = dist._func_dict['logpdf'](theta[idx,:], phi[key])
                logpdf.append(logpdf_i)
            return jnp.stack(logpdf, axis=0)

        def pdf(theta, phi):
            coeffs = coefficients(phi)
            pdf_comps = jnp.exp(logpdf_components(theta, phi))
            return jnp.sum(coeffs*pdf_comps).squeeze()

        def logpdf(theta, phi):
            return jnp.log(pdf(theta, phi))

        #
        #   Transformation functions:
        #

        def transform(epsilon, phi):
            theta = []
            for idx, (key, dist) in enumerate(components.items()):
                theta_i = dist._func_dict['transform'](epsilon[idx,:], phi[key])
                theta.append(theta_i)
            return jnp.stack(theta, axis=0)

        #
        #   Sampling functions:
        #

        def sample_base(num_samples, prngkey):
            epsilon = []
            for dist in approx_list:
                epsilon_i = dist._func_dict['sample_base'](num_samples, prngkey)
                epsilon.append(epsilon_i)
            return jnp.stack(epsilon, axis=1)

        def sample_components(prngkey, num_samples, num_choices, probs):
            return jax.random.choice(prngkey, num_choices, shape=(num_samples,), p=probs) 

        #
        #   Post-Processing:
        #

        jaxfunc_dict = cls._place_jaxfuncs_in_dict(coefficients, logpdf, logpdf_components, transform, sample_base, sample_components)
        jaxfunc_dict = cls._differentiate_jaxfuncs(jaxfunc_dict)
        jaxfunc_dict = cls._vectorise_jaxfuncs(jaxfunc_dict)

        return jaxfunc_dict

    #
    #   Parameter/Phi Methods
    #

    def phi(self, x=None):
        phi = {}
        for idx, approx in self._components.items():
            phi[self.component_key(idx)] = approx.phi()
        # Add batch dimension:
        phi[self._coeff_key] = self._log_unnorm_coeffs[None,:]
        return Jaxtainer(phi)

    def update(self, new_phi):
        for key in self._components.keys():
            self._components[key].update(new_phi[key])
        self._log_unnorm_coeffs = jnp.squeeze(new_phi[self._coeff_key])

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return len(self._components)

class Identical(MixtureApproximation):

    def __init__(self, approxdist, num):
        self._approxdist = approxdist
        self._num_components = num
        self._log_unnorm_coeffs = jnp.zeros(self._num_components)
        self._noncoeff_phi = Jaxtainer({self.component_key(idx): approxdist.phi() for idx in range(num)})
        self._jaxfunc_dict = self._create_jax_functions(self._approxdist, self._noncoeff_phi, self._num_components)
        super().__init__()
        
    @classmethod
    def _create_jax_functions(cls, approxdist, noncoeff_phi, num_components):

        #
        #   Coefficient functions:
        #

        def coefficients(phi):
            unnorm_coeffs = jnp.exp(phi[cls._coeff_key])
            norm_coeffs = unnorm_coeffs/jnp.sum(unnorm_coeffs)
            return norm_coeffs.squeeze()

        def stack_noncoeff_phi(phi):
            # Use np here since phi[key] is an arraytainer:
            return np.stack([phi[key] for key in noncoeff_phi.keys()], axis=0)

        #
        #   Probability density functions:
        #

        def preprocess_theta(theta):
            if theta.ndim < 2:
                theta = theta[None,:]
                theta = jnp.repeat(theta, repeats=num_components, axis=0)
            return theta

        # Function which computes logpdf wrt each component, vectorised over the num_mixture dimension:
        vmap_logpdf_components = \
        jax.vmap(lambda theta, noncoeff_phi : approxdist._func_dict['logpdf'](theta, noncoeff_phi), in_axes=(0,0))

        def logpdf_components(theta, phi):
            theta = preprocess_theta(theta)
            noncoeff_phi = stack_noncoeff_phi(phi)
            return vmap_logpdf_components(theta, noncoeff_phi)

        def pdf(theta, phi):
            coeffs = coefficients(phi)
            noncoeff_phi = stack_noncoeff_phi(phi)
            theta = preprocess_theta(theta)
            logpdf_comps = vmap_logpdf_components(theta, noncoeff_phi)
            pdf_comps = jnp.exp(logpdf_comps)
            pdf = jnp.sum(coeffs*pdf_comps).squeeze()
            return pdf

        def logpdf(theta, phi):
            lpdf = jnp.log(pdf(theta, phi))
            return lpdf

        #
        #   Transformation functions:
        #

        # Vectorise transform function over mixture components:
        vmap_transform = jax.vmap(approxdist._func_dict['transform'], in_axes=(0,0))

        def transform(epsilon, phi):
            noncoeff_phi = stack_noncoeff_phi(phi)
            return vmap_transform(epsilon, noncoeff_phi)

        #
        #   Sampling functions:
        #

        def sample_base(num_samples, prngkey):
            epsilon = approxdist._func_dict['sample_base'](num_samples, prngkey)
            return jnp.repeat(epsilon, num_components, axis=0)

        def sample_components(prngkey, num_samples, num_choices, probs):
            return jax.random.choice(prngkey, num_choices, shape=(num_samples,), p=probs) 

        #
        #   Post-Processing:
        #

        jaxfunc_dict = cls._place_funcs_in_dict(coefficients, logpdf, logpdf_components, transform, sample_base, sample_components)
        jaxfunc_dict = cls._differentiate_jaxfuncs(jaxfunc_dict)
        jaxfunc_dict = cls._vectorise_jaxfuncs(jaxfunc_dict)

        return jaxfunc_dict


    def phi(self, x=None):
        phi = self._noncoeff_phi.copy()
        phi[self._coeff_key] = self._log_unnorm_coeffs[None,:]
        return Jaxtainer(phi)

    def update(self, new_phi):
        for key in self._noncoeff_phi.keys():
            self._noncoeff_phi[key] = new_phi[key]
        self._log_unnorm_coeffs = jnp.squeeze(new_phi[self._coeff_key])

    @property
    def components(self):
        return {key: self._approxdist for key in self._noncoeff_phi.keys()}

    @property
    def num_components(self):
        return self._num_components