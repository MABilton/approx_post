import jax
import jax.numpy
from arraytainers import Jaxtainer
# Only need Gaussian import for GaussianMixture convenience class:
import approx

class MixtureApproximation:

    _coeff_key = 'unnorm_coeff'
    _unnorm_coeff_bounds = {'lb': 1e-2, 'ub': 1e2}
    
    def __init__(self, approxdists, coefficients=None):
        if isinstance(approxdists, list):
            approxdists = {idx: dist for idx, dist in enumerate(approxdists)}
        #create names list
        if self._coeff_key in approxdists:
            raise KeyError(f"Component not allowed to be named {self._coeff_key}.")
        self._components = approxdists
        self.unnorm_coefficients = self._initialise_coefficients(coefficients)
        self._coefficients, self._coefficients_del_phi = self._create_coefficients_funcs()

    def _initialise_coefficients(self, coefficients):
        if coefficients is None:
            coefficients = jnp.ones(self.num_components)
        else:
            coefficients = jnp.array(coefficients)
        return self._normalise_coefficients(coefficients)

    def _create_coefficients_funcs(self):

        def coefficients(phi):
            unnorm_coeffs = phi[self._coeff_key]
            return unnorm_coeffs/jnp.sum(unnorm_coeffs)
             
        coefficients_del_phi = jax.jacfwd(coefficient)
    
        return coefficients, coefficients_del_phi

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
        return self._coefficients(phi)

    def coefficients_del_phi(self, phi=None):
        phi = self._get_phi(phi)
        return self._coefficients_del_phi(phi)

    #
    #   Parameter/Phi Methods
    #

    def phi(self, x=None):
        phi = {}
        for key, approx in self._components.items():
            phi[key] = approx.phi()
        phi[self._coeff_key] = self.unnorm_coefficients
        return Jaxtainer(phi)

    def update(self, new_phi):
        for key in self._components.keys():
            self._components[key].update(new_phi[key])
        self.unnorm_coefficients = new_phi[self._coeff_key]

    @property
    def phi_bounds(self):
        phi_bounds = {'lb': {}, 'ub': {}}
        for key, approx in self._components.items():
            for b in ('lb', 'ub'):
                phi_bounds[b][key] = approx.phi_bounds[b]
        for b in ('lb', 'ub'):
            phi_bounds[b][self._coeff_key] = self._unnorm_coeff_bounds[b]*jnp.ones(self.num_components)
        return Jaxtainer(phi_bounds)

    #
    #   Log-Probability and Sampling Methods
    #

    def logpdf_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        logprob = []
        for approx in self._components.values():
            logprob.append(approx.logpdf(theta, phi))
        return jnp.stack(logprob, axis=0)

    def pdf_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        return jnp.exp(self.logpdf_components(theta, phi))

    def pdf(self, theta, phi=None):
        phi = self._get_phi(phi)
        return jnp.sum(coeffs*self.pdf_components(theta, phi), axis=0)

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
        component_samples = jax.random.choice(prngkey, jnp.array(range(self.num_components)), 
                                              p=jnp.append(self.coefficients.values()), shape=(num_samples,))
        for idx, dist in enumerate(self._components.values()):
            num_samples_i = jnp.sum(component_samples==idx)
            samples.append(dist.sample(num_samples_i, phi))
        samples = jnp.append(samples, axis=0)
        return jax.random.shuffle(prngkey, samples, axis=)

    def sample_base(self, num_samples, prngkey):
        phi = self._get_phi(phi)
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

class GaussianMixture(MixtureApproximation):

    def __init__(self, ndim, num_components, phi=None, mean_bounds=(None,None), var_bounds=(None,None), 
                 cov_bounds=(None,None), coefficients=None):
        components = []
        for i in range(num_components):
            components.append(approx.Gaussian(ndim, phi, mean_bounds, var_bounds, cov_bounds))
        super().__init__(components, coefficients)
