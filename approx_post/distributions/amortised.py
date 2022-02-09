import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
from arraytainers import Jaxtainer

class AmortisedApproximation:

    def __init__(self, distribution, phi_func, params):
        self.distribution = distribution
        self._phi_func = jax.vmap(phi_func, in_axes=(0,None))
        self._phi_del_params = jax.vmap(jax.jacfwd(phi_func, argnums=1), in_axes=(0,None))
        self.params = Jaxtainer(params)
    
    # See: https://stackoverflow.com/questions/26467564/how-to-copy-all-attributes-of-one-python-object-to-another
    def __getattr__(self, name):
        if name in dir(self):
            attr = getattr(self, name)
        else:
            attr = getattr(self.distribution, name)
        return attr

    def phi(self, x):
        for _ in range(2-x.ndim):
            x = x[None,:]
        return self._phi_func(x, self.params)
    
    def phi_del_params(self, x):
        for _ in range(2-x.ndim):
            x = x[None,:]
        return self._phi_del_params(x, self.params)
    
    def _get_phi(self, phi, x):
        if (phi is None) or (x is None):
            TypeError('Must specify either a phi value or an x value.')
        if phi is None:
            phi = self.phi(x)
        return phi

    # Methods used by all approximate distributions:
    def logpdf(self, theta, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.logpdf(theta, phi)
    
    def sample(self, num_samples, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.sample(num_samples, phi)

    def transform(self, epsilon, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.transform(epsilon, phi)
    
    def transform_del_2(self, epsilon, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.transform_del_2(epsilon, phi)

    def logpdf_del_1(self, theta, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.logpdf_del_1(theta, phi)
    
    def logpdf_del_2(self, theta, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.logpdf_del_2(theta, phi)

    # Methods used by mixture approximations:
    def coefficients(self, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.coefficients(phi)
    
    def logprob_components(self, theta, phi=None, x=None):
        phi = self._get_phi(phi, x)
        self.distribution.logprob_components(theta, phi)

    def logprob_del_1_components(self, phi=None, x=None):
        phi = self._get_phi(phi, x)
        self.distribution.logprob_del_1_components(theta, phi)

    def update(self, new_params):
        self.params = Jaxtainer(new_params)

    def save(self):
        pass
    
    def load(self):
        pass

class LinearRegression(AmortisedApproximation):

    def __init__(self, distribution, x_dim, order):

    def _create_regression_func(self, distribution, x_dim):


class NeuralNetwork(AmortisedApproximation):

    _activations = {'relu': jnn.relu, 'sigmoid': jnn.sigmoid}
    _initializers = {'W': jnn.initializers.he_normal(), 'b': jnn.initializers.zeros}

    def __init__(self, distribution, x_dim, prngkey, num_layers=5, width=5, activation='relu'):
        
        try:
            assert activation.lower() in self._activations
        except (AssertionError, AttributeError):
            raise ValueError(f'Invalid value specified for activation; valid options are: {", ".join(self._activations.keys())}')

        nn_layers, wts = self._create_nn_layers(distribution, x_dim, num_layers, width, activation, prngkey)
        nn = self._create_nn_func(nn_layers, num_layers)
        super().__init__(distribution, phi_func=nn, params=wts)

    def _create_nn_layers(self, distribution, x_dim, num_layers, width, activation, prngkey):
        
        phi_dim = distribution.phi().size

        nn_layers, wts = {}, {}
        for layer in range(num_layers+2):
            # Input layer:
            if layer == 0:
                nn_layers[layer] = self._create_layer_func(self._activations[activation]) 
                W, b = self._initialise_layer_params(prngkey, in_dim=x_dim, out_dim=width)
            # Intermediate layer:
            elif layer < num_layers+1:
                nn_layers[layer] = self._create_layer_func(self._activations[activation]) 
                W, b = self._initialise_layer_params(prngkey, in_dim=width, out_dim=width)
            # Output layer:
            else:
                nn_layers[layer] = self._create_layer_func(self._activations['sigmoid']) 
                W, b = self._initialise_layer_params(prngkey, in_dim=width, out_dim=phi_dim)
            wts[f'W_{layer}'], wts[f'b_{layer}'] = W, b
            # Update prngkey so that each layer is initialised differently:
            prngkey = jax.random.split(prngkey, num=1).squeeze()

        # Don't select first batch dimension in phi shape:
        phi_shape = distribution.phi().shape[1:]
        nn_layers[num_layers+2] = self._create_postprocessing_func(phi_shape, distribution.phi_bounds)

        return nn_layers, wts

    @staticmethod
    def _create_layer_func(activation):
        def layer(x, W, b):
            # NN function will be vectorised over batch dimension of x:
            return activation(jnp.einsum('ij,i->j', W, x) + b)
        return layer

    def _initialise_layer_params(self, prngkey, in_dim, out_dim):
        W = self._initializers['W'](prngkey, (in_dim, out_dim))
        b = self._initializers['b'](prngkey, (out_dim,))
        return W, b
    
    @staticmethod
    def _create_postprocessing_func(phi_shape, phi_bounds):
        def postprocessing_fun(x):
            x = Jaxtainer.from_array(x, shapes=phi_shape)
            return (phi_bounds['ub'] - phi_bounds['lb'])*x + phi_bounds['lb']
        return postprocessing_fun
    
    @staticmethod
    def _create_nn_func(nn_layers, num_layers):
        def nn(x, params):
            for layer in range(num_layers+2):
                x = nn_layers[layer](x, params[f'W_{layer}'], params[f'b_{layer}'])
            x = nn_layers[num_layers+2](x)
            return x
        return nn