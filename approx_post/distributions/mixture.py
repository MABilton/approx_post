import math
import copy
import numpy as np
import jax
import jax.numpy as jnp
from arraytainers import Jaxtainer

class MixtureApproximation:

    _coeff_key = 'log_unnorm_coeff'

    #
    #   Function Creation Methods
    #

    @classmethod
    def _create_shared_funcs(cls, num_components):

        def coefficients(phi):
            unnorm_coeffs = jnp.exp(phi[cls._coeff_key])
            norm_coeffs = unnorm_coeffs/jnp.sum(unnorm_coeffs)
            return jnp.atleast_1d(norm_coeffs.squeeze())

        def sample_idx(prngkey, num_samples, num_choices, probs):
            return jax.random.choice(prngkey, num_choices, shape=(num_samples,), p=probs) 

        def add_mixture_dim(theta_or_epsilon):
            if theta_or_epsilon.ndim < 2:
                theta_or_epsilon = theta_or_epsilon[None,:]
                theta_or_epsilon = jnp.repeat(theta_or_epsilon, repeats=num_components, axis=0)
            return theta_or_epsilon

        return coefficients, sample_idx, add_mixture_dim
        
    @staticmethod
    def _create_pdf_funcs(coefficients, logpdf_components, transform):

        def pdf(theta, phi):
            coeffs = coefficients(phi)
            pdf_comps = jnp.exp(logpdf_components(theta, phi))
            return jnp.sum(coeffs*pdf_comps).squeeze()

        def logpdf(theta, phi):
            return jnp.log(pdf(theta, phi))

        # Vectorise logpdf over mixture dimension of theta:
        logpdf_vmap = jax.vmap(logpdf, in_axes=(0,None))
        def logpdf_epsilon(epsilon, phi):
            theta = transform(epsilon, phi) # shape = (num_mixture, theta_dim)
            return logpdf_vmap(theta, phi)

        return pdf, logpdf, logpdf_epsilon

    @classmethod
    def _vectorise_and_differentiate_funcs(cls, **func_dict):
        jaxfunc_dict = cls._differentiate_jaxfuncs(func_dict)
        jaxfunc_dict = cls._vectorise_jaxfuncs(jaxfunc_dict)
        return jaxfunc_dict

    @staticmethod
    def _differentiate_jaxfuncs(jaxfunc_dict):
        
        grad_funcs = {}

        for key, func in jaxfunc_dict.items():

            if key == 'coefficients':
                grad_funcs['coefficients_del_phi'] = jax.jacfwd(func, argnums=0)

            elif 'epsilon' in key:
                grad_funcs['logpdf_epsilon_del_2'] = jax.jacfwd(func, argnums=1)
            
            # Differentiate 'pdf' and 'logpdf' functions:
            elif 'pdf' in key:
                grad_funcs[f'{key}_del_1'] = jax.jacfwd(func, argnums=0)
                grad_funcs[f'{key}_del_2'] = jax.jacfwd(func, argnums=1)

            elif key == 'transform':
                grad_funcs['transform_del_2'] = jax.jacfwd(func, argnums=1)

        return {**jaxfunc_dict, **grad_funcs}

    @staticmethod
    def _vectorise_jaxfuncs(jaxfunc_dict):
        
        for key, func in jaxfunc_dict.items():
            
            # Vectorise 'coefficients' and 'coefficients_del_phi' over batch dimension of phi input:
            if 'coefficients' in key:
                jaxfunc_dict[key] = jax.vmap(func, in_axes=0)

            # Vectorise over batch dimension of phi, as well as over sample dimension of epsilon input:
            elif 'epsilon' in key:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None)), in_axes=(None,0))

            # Vectorise over batch dimension of theta and phi inputs, as well as over sample dimension of theta input:
            elif 'pdf' in key:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None)), in_axes=(0,0))
            
            # Vectorise 'transform' and 'transform_del_2' over batch dimension of phi, and over sample dimension of epsilon:
            elif 'transform' in key:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None)), in_axes=(None,0))
            
            # Vectorise over batch dimension of probs:
            elif key == 'sample_idx':
                jaxfunc_dict[key] = jax.vmap(func, in_axes=(None,None,None,0))
        
        return jaxfunc_dict

    #
    #   Pre-Processing and Post-Processing Methods
    #

    def _get_phi(self, phi):
        if phi is None:
            phi = self.phi()
        return phi

    def _preprocess_theta(self, theta, phi):

        # Adds num_batch and num_samples to theta if not specified

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
            raise ValueError(f'Batch dimension of theta (= {theta_batch_dim}) and phi (= {phi_batch_dim}) must match.')  
        
        return theta        

    #
    #   Parameter Methods
    #

    @staticmethod
    def component_key(idx):
        return f'component_{idx}'

    def phi(self, x=None):
        # Add batch dimension to params (defined in child classes)
        return self.params[None,:]

    def coefficients(self, phi=None):
        phi = self._get_phi(phi)
        return self._jaxfunc_dict['coefficients'](phi)
    
    def coefficients_del_phi(self, phi=None):
        phi = self._get_phi(phi)
        coeffs_del_phi = self._jaxfunc_dict['coefficients_del_phi'](phi)
        return coeffs_del_phi

    #
    #   Log-Probability and Sampling Methods
    #

    def pdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        pdf = self._jaxfunc_dict['pdf'](theta, phi)
        num_batch, num_samples = theta.shape[:2]
        return pdf.reshape(num_batch, num_samples)

    def logpdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        logpdf = self._jaxfunc_dict['logpdf'](theta, phi)
        return logpdf

    def logpdf_del_1(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        logpdf_del_1 = self._jaxfunc_dict['logpdf_del_1'](theta, phi)
        return logpdf_del_1.reshape(*theta.shape[:-1], theta_dim)

    def logpdf_del_2(self, theta, phi=None):
        phi = self._get_phi(phi)
        theta = self._preprocess_theta(theta, phi)
        logpdf_del_2 = self._jaxfunc_dict['logpdf_del_2'](theta, phi)
        num_batch, num_samples = theta.shape[0:2]
        return logpdf_del_2.reshape(num_batch, num_samples, phi.shape[1:])

    def logpdf_epsilon(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        logpdf = self._jaxfunc_dict['logpdf_epsilon'](epsilon, phi)
        num_samples, num_components = epsilon.shape[:-1]
        return logpdf.reshape(-1, num_samples, num_components)

    def logpdf_epsilon_del_2(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        logpdf_del_2 = self._jaxfunc_dict['logpdf_epsilon_del_2'](epsilon, phi)
        num_samples, num_components = epsilon.shape[:-1]
        return logpdf_del_2.reshape(-1, num_samples, num_components, phi.shape[1:])

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
        num_samples, num_components, dim_theta = epsilon.shape
        return theta_del_2.reshape(-1, num_samples, num_components, dim_theta, phi.shape[1:])

    #
    #   Sample Methods
    #

    def sample_idx(self, num_samples, prngkey, phi=None):
        phi = self._get_phi(phi)
        probs = self.coefficients(phi)
        num_choices = self.num_components
        return self._jaxfunc_dict['sample_idx'](prngkey, num_samples, num_choices, probs)

    def sample_base(self, num_samples, prngkey):
        epsilon = self._jaxfunc_dict['sample_base'](num_samples, prngkey)
        theta_dim = epsilon.shape[-1]
        return epsilon.reshape(num_samples, self.num_components, theta_dim)
    
    def sample(self, num_samples, prngkey, phi=None):
        
        phi = self._get_phi(phi)

        idx_samples = self.sample_idx(num_samples, prngkey, phi) # shape = (num_batch, num_samples)

        samples = []
        for idx, (key, dist) in enumerate(self.components.items()):
            num_samples_i = jnp.sum(idx_samples == idx, axis=1) # shape = (num_batch,)
            # Can't vectorise over num_samples input - draw maximum number of samples requested across all batches
            # for each batch instead:
            max_num_samples_i = jnp.max(num_samples_i)
            samples_i = dist.sample(max_num_samples_i, prngkey, phi[key]) # shape = (num_batch, max_num_samples_i, theta_dim)
            # For phi values where we've drawn too many samples:
            samples_i = self._replace_excess_samples_with_nan(samples_i, num_samples_i)
            samples.append(samples_i)

        samples = self._combine_samples_from_each_dist(samples, num_samples, prngkey)

        return samples

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
    def _combine_samples_from_each_dist(samples, num_samples, prngkey):
        # Append samples taken from different components along samples axis:
        samples = jnp.concatenate(samples, axis=1) # shape = (num_batch, sum(max(num_samples_i)), theta_dim)
        not_nan_idx = jnp.logical_not(jnp.isnan(samples))
        samples_without_nan = samples.at[not_nan_idx].get()
        num_batch, theta_dim = samples.shape[0], samples.shape[-1]
        samples = samples_without_nan.reshape(num_batch, num_samples, theta_dim)
        return jax.random.permutation(prngkey, samples, axis=1)

class Different(MixtureApproximation):

    def __init__(self, components):
        self._components = {self.component_key(idx): copy.deepcopy(dist) for idx, dist in enumerate(components)}
        self._log_unnorm_coeffs = jnp.zeros(self.num_components)
        self._jaxfunc_dict = self._create_jaxfunc_dict(self._components)

    @classmethod
    def _create_jaxfunc_dict(cls, components):

        # Deepcopy so that changes to components doesn't break funcs:
        components = copy.deepcopy(components)

        coefficients, sample_idx, add_mixture_dim = super()._create_shared_funcs(num_components=len(components))

        def logpdf_components(theta, phi):
            theta = add_mixture_dim(theta)
            logpdf = []
            for idx, (key, dist) in enumerate(components.items()):
                logpdf_i = dist._func_dict['logpdf'](theta[idx,:], phi[key])
                logpdf.append(logpdf_i)
            return jnp.stack(logpdf, axis=0)

        def transform(epsilon, phi):
            epsilon = add_mixture_dim(epsilon)
            theta = []
            for idx, (key, dist) in enumerate(components.items()):
                theta_i = dist._func_dict['transform'](epsilon[idx,:], phi[key])
                theta.append(theta_i)
            return jnp.stack(theta, axis=0)

        def sample_base(num_samples, prngkey):
            epsilon = []
            for dist in components.values():
                epsilon_i = dist._func_dict['sample_base'](num_samples, prngkey)
                epsilon.append(epsilon_i)
            return jnp.stack(epsilon, axis=0)

        pdf, logpdf, logpdf_epsilon = super()._create_prob_funcs(coefficients, logpdf_components, transform)

        jax_func_dict = super()._vectorise_and_differentiate_funcs(coefficients=coefficients, pdf=pdf,               
                                logpdf=logpdf, logpdf_epsilon=logpdf_epsilon,  transform=transform, 
                                sample_base=sample_base, sample_idx=sample_idx)

        return jax_func_dict

    #
    #   Component Methods
    #

    def add_component(self, new_component):
        new_key = self.component_key(self._num_components-1)
        self._components[new_key] = copy.deepcopy(new_component)
        self._log_unnorm_coeffs = jnp.append(self._log_unnorm_coeffs, jnp.mean(self._log_unnorm_coeffs))
        self._jaxfunc_dict = self._create_jaxfunc_dict(self._components)

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return len(self._components)

    #
    #   Parameter Methods
    #

    @property
    def params(self):
        phi = {}
        for key, approx in self._components.items():
            phi[key] = approx.params
        # Add batch dimension:
        phi[self._coeff_key] = self._log_unnorm_coeffs
        return Jaxtainer(phi)

    def perturb(self, perturb_dict):
        self._log_unnorm_coeffs += perturb_dict[self._coeff_key]
        for key in self.components.keys():
            perturb_params = self._components[key].params + perturb_dict[key]
            self._components[key].update(perturb_params)

    def update(self, new_phi):
        for key in self.components.keys():
            self._components[key].update(new_phi[key])
        self._log_unnorm_coeffs = jnp.atleast_1d(new_phi[self._coeff_key].squeeze())

class Identical(MixtureApproximation):

    #
    #   Constructor Methods
    #

    def __init__(self, approxdist, num_components):
        self._approxdist = copy.deepcopy(approxdist)
        self._num_components = num_components
        self._log_unnorm_coeffs, self._noncoeff_phi = self._initialise_phi(approxdist, num_components)
        self._jaxfunc_dict = self._create_jaxfunc_dict(self._approxdist, self._noncoeff_phi, num_components)

    @classmethod
    def _initialise_phi(cls, approxdist, num_components):
        log_unnorm_coeffs = jnp.zeros(num_components)
        noncoeff_phi = Jaxtainer({cls.component_key(idx): copy.deepcopy(approxdist.params) for idx in range(num_components)})
        return log_unnorm_coeffs, noncoeff_phi

    @classmethod
    def _create_jaxfunc_dict(cls, approxdist, approxdist_noncoeff_phi, num_components):

        # Deepcopy so that changes to approxdist and approxdist_noncoeff_phi doesn't break funcs:
        approxdist, approxdist_noncoeff_phi = copy.deepcopy(approxdist), copy.deepcopy(approxdist_noncoeff_phi)
        
        coefficients, sample_idx, add_mixture_dim = super()._create_shared_funcs(num_components)

        # Stacks noncoeff phi values into a single matrix:
        stack_noncoeff_phi = lambda phi: np.stack([phi[key] for key in approxdist_noncoeff_phi.keys()], axis=0)

        # Vectorised over the num_mixture dimension of theta and phi:
        vmap_logpdf_components = \
        jax.vmap(lambda theta, noncoeff_phi : approxdist._func_dict['logpdf'](theta, noncoeff_phi), in_axes=(0,0))

        def logpdf_components(theta, phi):
            # Use np here since phi[key] is an arraytainer:
            noncoeff_phi = stack_noncoeff_phi(phi)
            theta = add_mixture_dim(theta)
            return vmap_logpdf_components(theta, noncoeff_phi)

        # Vectorised over the num_mixture dimension of epsilon and phi:
        vmap_transform = jax.vmap(approxdist._func_dict['transform'], in_axes=(0,0))
        
        def transform(epsilon, phi):
            noncoeff_phi = stack_noncoeff_phi(phi)
            epsilon = add_mixture_dim(epsilon)
            return vmap_transform(epsilon, noncoeff_phi)

        def sample_base(num_samples, prngkey):
            epsilon = approxdist._func_dict['sample_base'](num_samples, prngkey)
            return jnp.repeat(epsilon, num_components, axis=0)

        pdf, logpdf, logpdf_epsilon = super()._create_pdf_funcs(coefficients, logpdf_components, transform)

        jax_func_dict = super()._vectorise_and_differentiate_funcs(coefficients=coefficients, transform=transform,           
                        logpdf=logpdf, logpdf_epsilon=logpdf_epsilon, pdf=pdf, sample_base=sample_base, sample_idx=sample_idx)

        return jax_func_dict

    #
    #   Component Methods
    #

    @property
    def components(self):
        return {key: self._approxdist for key in self._noncoeff_phi.keys()}

    @property
    def num_components(self):
        return self._num_components

    def add_component(self):
        self._num_components += 1
        # Append mean of coefficients:
        self._log_unnorm_coeffs = jnp.append(self._log_unnorm_coeffs, jnp.mean(self._log_unnorm_coeffs))
        # Add parameters for new component:
        new_key = self.component_key(self._num_components-1)
        self._noncoeff_phi[new_key] = copy.deepcopy(self._approxdist.params)
        # Update function dictionary:
        self._jaxfunc_dict = self._create_jaxfunc_dict(self._approxdist, self._noncoeff_phi, self._num_components)

    #
    #   Parameter Methods
    #

    @property
    def params(self):
        phi = self._noncoeff_phi.copy()
        phi[self._coeff_key] = self._log_unnorm_coeffs
        return Jaxtainer(phi)

    def update(self, new_phi):
        for key in self._noncoeff_phi.keys():
            self._noncoeff_phi[key] = new_phi[key]
        self._log_unnorm_coeffs = jnp.atleast_1d(new_phi[self._coeff_key].squeeze())

    def perturb(self, perturb_dict):
        self._log_unnorm_coeffs += perturb_dict[self._coeff_key]
        for key in self._noncoeff_phi.keys():
            self._noncoeff_phi[key] += perturb_dict[key]