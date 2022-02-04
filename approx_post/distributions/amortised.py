import jax
import jax.numpy as jnp
import jax.nn as jnn
from arraytainers import Jaxtainer

class AmortisedApproximation:

    def __init__(self, distribution, phi_func, params):
        self.distribution = distribution
        self._phi_func = phi_func
        self._phi_del_params = jax.vmap(jax.jacfwd(phi_func, argnums=1), in_axes=(0,0))
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
        phi = self._phi_func(x, self.params)
        return self._constraint(phi)
    
    def phi_del_params(self, x):
        for _ in range(2-x.ndim):
            x = x[None,:]
        return self._phi_del_params(x, self.params)
    
    def _get_phi(self, phi, x):
        if all(None in [phi, x]):
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
        return self.distribution.transform(num_samples, phi)
    
    def transform_del_2(self, epsilon, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.transform_del_2(num_samples, phi)

    def logpdf_del_1(self, theta, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.logpdf_del_1(num_samples, phi)
    
    def logpdf_del_2(self, theta, phi=None, x=None):
        phi = self._get_phi(phi, x)
        return self.distribution.logpdf_del_2(num_samples, phi)

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

    def update(self, new_params)
        self.params = Jaxtainer(new_params)

    def save(self):
        pass
    
    def load(self):
        pass

class NeuralNetwork(AmortisedApproximation):

    _activations = {'relu': jnn.relu, 'softmax': jnn.softmax}
    _initializers = {'W': jnn.initializers.he_normal(), 'b': jnn.initializers.zeros}

    def __init__(self, distribution, x_dim, prngkey, num_layers=5, width=10, activation='relu'):
        
        try:
            assert activation.lower() in self._activations
        except AssertionError, AttributeError:
            raise ValueError(f"Invalid value specified for activation; valid options are: {", ".join(self._activations.keys())}")

        nn_layers, wts = self._create_nn_layers(distribution, x_dim, num_layers, width, activation, prngkey)
        nn = self._create_nn_func(nn_layers, num_layers)
        super().__init__(distribution, phi_func=nn, params=wts)

    def _create_nn_layers(self, distribution, x_dim, num_layers, width, activation, prngkey):
        
        phi_shape = distribution.phi().shape
        phi_dim = phi_shape.sum_all().item()

        nn_layers, wts = {}, {}
        for layer in range(num_layers+2):
            # Input layer:
            if layer == 0:
                nn_layers[layer] = self._create_layer_func(self._activations[activation]) 
                wts[f'W_{layer}'], wts[f'b_{layer}'] = self._initialise_layer_params(prngkey, in_dim=x_dim, out_dim=width)
            # Intermediate layer:
            elif layer < num_layers+1:
                nn_layers[layer] = self._create_layer_func(self._activations[activation]) 
                wts[f'W_{layer}'], wts[f'b_{layer}'] = self._initialise_layer_params(prngkey, in_dim=width, out_dim=width)
            # Output layer:
            else:
                nn_layers[layer] = self._create_layer_func(self._activations['softmax']) 
                wts[f'W_{layer}'], wts[f'b_{layer}'] = self._initialise_layer_params(prngkey, in_dim=x_dim, out_dim=phi_dim)
        
        # Create function which reshapes NN's output vector to a Jaxtainer of the correct shape:
        nn_layers[num_layers+2] = self._create_postprocessing_func(phi_shape, distribution.phi_bounds)

        return nn_layers, wts

    @staticmethod
    def _create_layer_func(activation):
        return lambda x, W, b : activation(jnp.einsum('ij,aj->ai', W, x) + b)

    def _initialise_layer_params(self, prngkey, in_dim, out_dim):
        W = self._initializers['W'](prngkey, out_dim)
        b = self._initializers['b'](prngkey, (in_dim, out_dim))
        return W, b
    
    @staticmethod
    def _create_postprocessing_func(phi_shape, phi_bounds):
        def postprocessing_fun(x):
            x = Jaxtainer.from_vector(x, phi_shape)
            return (phi_bounds['ub'] - phi_bounds['lb'])*x + phi_bounds['lb']
        return postprocessing_fun
    
    @staticmethod
    def _create_nn_func(nn_layers, num_layers):
        def nn(x, params):
            for layer in range(num_layers+2):
                x = nn_layers[layer](x, params[f'W_{layer}'], params[f'b_{layer}'])
           return nn_layers[num_layers+2](x)
        return nn