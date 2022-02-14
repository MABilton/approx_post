import jax
import jax.numpy as jnp
from arraytainers import Jaxtainer

class MixtureApproximation:

    _coeff_key = 'log_unnorm_coeff'
    # Vectorise over batch dimension of array and p argument:
    _randomly_choose_component_samples = jax.vmap(jax.random.choice, in_axes=(None,1,0,None,None), out_axes=0)
    
    def __init__(self, approxdists):
        if isinstance(approxdists, list):
            approxdists = {idx: dist for idx, dist in enumerate(approxdists)}
        if self._coeff_key in approxdists:
            raise KeyError(f"Component not allowed to be named {self._coeff_key}.")
        self._components = approxdists
        self._log_unnorm_coeffs = jnp.zeros(self.num_components)
        self._coeff_funcs = self._create_coeff_funcs()

    def _create_coeff_funcs(self):

        def coeffs(phi):
            unnorm_coeffs = jnp.exp(phi[self._coeff_key])
            norm_coeffs = unnorm_coeffs/jnp.sum(unnorm_coeffs)
            return norm_coeffs.squeeze()
             
        coeffs_del_phi = jax.vmap(jax.jacfwd(coeffs), in_axes=0)
    
        return {'coeffs': jax.vmap(coeffs, in_axes=0),
                'coeff_del_phi': coeffs_del_phi}

    #
    #   Coefficient Methods
    #

    @property
    def num_components(self):
        return len(self._components)

    def _get_phi(self, phi):
        if phi is None:
            phi = self.phi()
        return phi

    def coefficients(self, phi=None):
        phi = self._get_phi(phi)
        return self._coeff_funcs['coeffs'](phi)

    def coefficients_del_phi(self, phi=None):
        phi = self._get_phi(phi)
        return self._coeff_funcs['coeff_del_phi'](phi)

    #
    #   Parameter/Phi Methods
    #

    def phi(self, x=None):
        phi = {}
        for idx, approx in self._components.items():
            phi[f'component_{idx}'] = approx.phi()
        # Add batch dimension:
        phi[self._coeff_key] = self._log_unnorm_coeffs[None,:]
        return Jaxtainer(phi)

    def update(self, new_phi):
        for key in self._components.keys():
            self._components[key].update(new_phi[key])
        self._log_unnorm_coeffs = new_phi[self._coeff_key]

    #
    #   Log-Probability and Sampling Methods
    #

    def logpdf_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        logprob = []
        for idx, approx in enumerate(self._components.values()):
            phi_i = phi[f'component_{idx}']
            logprob.append(approx.logpdf(theta, phi_i))
        return jnp.stack(logprob, axis=0)

    def pdf_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        return jnp.exp(self.logpdf_components(theta, phi))

    def pdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        coeffs = self.coefficients(phi)
        return jnp.einsum('am,mab->ab', coeffs, self.pdf_components(theta, phi))

    def logpdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        return jnp.log(self.pdf(theta, phi))
    
    def logpdf_del_1_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        logpdf_del_1 = []
        for approx in self._components.values():
            logpdf_del_1.append(approx.logpdf_del_1(theta, phi))
        return jnp.stack(logpdf_del_1, axis=0)

    def pdf_del_1_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        logpdf_del_1_components = self.logpdf_del_1_components(theta, phi)
        pdf_components = self.pdf(theta, phi)
        return jnp.einsum('m,m...->m...', pdf_components, logpdf_del_1_components)

    def logpdf_del_1(self, theta, phi=None):
        phi = self._get_phi(phi)
        pdf_components = self.pdf_components(theta, phi)
        pdf_del_1_components = self.pdf_del_1_components(theta, phi)
        coeffs = self.coefficients(phi)
        return jnp.einsum('m,m...->m...', pdf_components, pdf_del_1_components)/jnp.sum(coeffs*pdf_components, axis=0)

    def logpdf_del_2_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        logpdf_del_2 = []
        for approx in self._components.values():
            logpdf_del_2.append(approx.logpdf_del_2(theta, phi))
        return jnp.stack(logpdf_del_2, axis=0)

    def pdf_del_2_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        logpdf_del_2_components = self.logpdf_del_2_components(theta, phi)
        pdf_components = self.pdf(theta, phi)
        return jnp.einsum('m,m...->m...', pdf_components, logpdf_del_2_components)

    def logpdf_del_2(self, theta, phi=None):
        phi = self._get_phi(phi)
        pdf_components = self.pdf_components(theta, phi)
        pdf_del_2_components = self.pdf_del_2_components(theta, phi)
        coeffs = self.coefficients(phi)
        coeffs_del_phi = self.coefficients_del_phi(phi)
        return (coeffs_del_phi*pdf_components + coeffs*pdf_del_2_components)/jnp.sum(coeffs*pdf_components, axis=0)

    def sample(self, num_samples, prngkey, phi=None):
        
        phi = self._get_phi(phi)
        samples = []
        for idx, dist in enumerate(self._components.values()):
            samples_i = dist.sample(num_samples, prngkey, phi[f'component_{idx}']) # shape = (num_batch, num_samples, theta_dim)
            samples.append(samples_i)

        samples = jnp.concatenate(samples, axis=0) # shape = (num_mixture, num_batch, num_samples, theta_dim)
        samples = self._randomly_choose_component_samples(prngkey, samples, p=self.coefficients(phi), replace=False, axis=0) 
        # shape = (num_batch, num_samples, theta_dim)

        return samples

    def sample_base(self, num_samples, prngkey):
        epsilon = []
        for approx in self._components.values():
            epsilon.append(approx.sample_base(num_samples, prngkey))
        return jnp.stack(epsilon, axis=0)

    def transform(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        theta = []
        for idx, approx in enumerate(self._components.values()):
            theta.append(approx.transform(epsilon[idx,:,:]))
        return jnp.stack(theta, axis=0)
    
    def transform_del_2(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        theta = []
        for idx, approx in enumerate(self._components.values()):
            theta.append(approx.transform(epsilon[idx,:,:]))
        return jnp.stack(theta, axis=0)

class Repeated(MixtureApproximation):

    def __init__(self, approxdist, num):
        self._num_components = num
        self._components = {key: approxdist for key in range(num)}
        self._phi = {f'component_{idx}': dist.phi() for idx, dist in self._components.items()}
        self._log_unnorm_coeffs = jnp.zeros(self.num_components)
        self._coeff_funcs = self._create_coeff_funcs()

    def phi(self, x=None):
        phi = self._phi
        phi[self._coeff_key] = self._log_unnorm_coeffs[None,:]
        return Jaxtainer(phi)

    def update(self, new_phi):
        for key in self._phi.keys():
            self._phi[key] = new_phi[key]
        self._log_unnorm_coeffs = new_phi[self._coeff_key]