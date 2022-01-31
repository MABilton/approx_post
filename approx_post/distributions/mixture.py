import jax
import jax.numpy
from arraytainers import Jaxtainer
from approx import Gaussian

class MixtureApproximation:

    _coeff_key = 'coeff'

    def __init__(self, approxdists, coefficients=None):
        if isinstance(approxdists, list):
            approxdists = {idx: dist for idx, dist in enumerate(approxdists)}
        if self._coeff_key in approxdists:
            raise KeyError(f"Component not allowed to be named {self._coeff_key}.")
        self.components = approxdists
        self._coefficients = self._initialise_coefficients(coefficients)
    
    def _initialise_coefficients(self, coefficients):
        if coefficients is None:
            coefficients = jnp.ones((self.num_components-1,))/self.num_components
        else:
            coefficients = jnp.array(coefficients)
            coefficients = coefficients[:-1]/jnp.sum(coefficients)
        return coefficients

    @property
    def num_components(self):
        return len(self.components)

    def phi(self, x=None):
        phi = {}
        for key, approx in self.components.items():
            phi[key] = approx.phi()
        phi[self._coeff_key] = self.coefficients()
        return Jaxtainer(phi)

    def coefficients(self, phi=None):
        if phi is None:
            all_but_last_coeff = self._coefficients
        else:
            all_but_last_coeff = self._get_coefficients(phi)
        last_coeff = 1 - jnp.sum(all_but_last_coeff)
        return jnp.append(all_but_last_coeff, last_coeff)

    def _get_coefficients(self, phi):
        return phi[self._coeff_key]

    def _get_phi(self, phi):
        if phi is None:
            phi = self.phi()
        return phi

    def logpdf_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        logprob = []
        for approx in self.components.values():
            logprob.append(approx.logpdf(theta, phi))
        return jnp.stack(logprob, axis=0)

    def pdf(self, theta, phi=None)
        phi = self._get_phi(phi)
        logprob_components = self.logprob_components(theta, phi)
        coeffs = self.coefficients(phi)
        return jnp.einsum('m...,m...', coeffs, jnp.exp(logprob_components))

    def logpdf(self, theta, phi=None):
        return jnp.log(self.pdf(theta, phi))
    
    def logpdf_del_1_components(self, theta, phi=None):
        phi = self._get_phi(phi)
        logprob_del_1 = []
        for approx in self.components.values():
            logprob_del_1.append(approx.logpdf_del_1(theta, phi))
        return jnp.stack(logprob_del_1, axis=0)

    def logpdf_del_1(self, theta, phi=None):
        phi = self._get_phi(phi)
        logprob_del_1_components = self.logprob_del_1_components(theta, phi)
        coeffs = self.coefficients(phi)
        return jnp.einsum('m...,m...', coeffs, jnp.exp(logprob_components))

    def sample(self, num_samples, prngkey, phi=None):
        phi = self._get_phi(phi)
        component_samples = jax.random.choice(prngkey, jnp.array(range(self.num_components)), 
                                              p=jnp.append(self.coefficients.values()), shape=(num_samples,))
        for idx, dist in enumerate(self.components.values()):
            num_samples_i = jnp.sum(component_samples==idx)
            samples.append(dist.sample(num_samples_i, phi))
        samples = jnp.append(samples, axis=0)
        return jax.random.shuffle(prngkey, samples, axis=)

    def sample_base(self, num_samples, prngkey):
        phi = self._get_phi(phi)
        epsilon = []
        for approx in self.components.values():
            epsilon.append(approx.sample_base(num_samples, prngkey))
        return jnp.stack(epsilon, axis=0)

    def transform(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        theta = []
        for idx, approx in enumerate(self.components.values()):
            theta.append(approx.transform(epsilon[idx,:,:]))
        return jnp.stack(theta, axis=0)
    
    def transform_del_2(self, epsilon, phi=None):
        phi = self._get_phi(phi)
        theta = []
        for idx, approx in enumerate(self.components.values()):
            theta.append(approx.transform(epsilon[idx,:,:]))
        return jnp.stack(theta, axis=0)

    def update(self, new_phi):
        for key in self.components.keys():
            self.components[key].update(new_phi[key])
        self._coefficients = new_phi[self._coeff_key]

    @property
    def phi_bounds(self):


class GaussianMixture(MixtureApproximation):

    def __init__(self, ndim, num_components, phi=None, mean_bounds=(None,None), var_bounds=(None,None), cov_bounds=(None,None),
                 coefficients=None):
        components = []
        for i in range(num_components):
            components.append(Gaussian(ndim, phi, mean_bounds, var_bounds, cov_bounds))
        self._components = components
        super().__init__(self._components)