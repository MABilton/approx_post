import itertools
import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
from arraytainers import Jaxtainer

class AmortisedApproximation:

    def __init__(self, distribution, phi_func, params, preprocessing=None):
        self.distribution = distribution
        self._phi_func = jax.vmap(phi_func, in_axes=(0,None))
        self._phi_del_params = jax.vmap(jax.jacfwd(phi_func, argnums=1), in_axes=(0,None))
        self.params = Jaxtainer(params)
        self.preprocessing = preprocessing
    
    # See: https://stackoverflow.com/questions/26467564/how-to-copy-all-attributes-of-one-python-object-to-another
    def __getattr__(self, name):
        if name in dir(self):
            attr = getattr(self, name)
        else:
            attr = getattr(self.distribution, name)
        return attr

    def _preprocess_x(self, x):
        for _ in range(2-x.ndim):
            x = x[None,:]
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        return x

    def phi(self, x):
        x = self._preprocess_x(x)
        return self._phi_func(x, self.params)
    
    def phi_del_params(self, x):
        x = self._preprocess_x(x)
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

    def __init__(self, distribution, x_dim, prngkey, order=1, preprocessing=None):
        lr_func, feature_func, params = self._create_regression_func(distribution, x_dim, order, prngkey)
        self.feature_func = jax.vmap(feature_func, in_axes=0)
        super().__init__(distribution, phi_func=lr_func, params=params, preprocessing=preprocessing)

    def _create_regression_func(self, distribution, x_dim, order, prngkey):

        # if isinstance(order, int):
        #     powers = Jaxtainer({key: jnp.arange(0,order) + 1 for key in distribution.phi().keys()}) 
        # else:
        #     powers = np.arange(0, Jaxtainer(order)) + 1

        powers = jnp.arange(0,order) + 1 

        phi_size = distribution.phi().size 
        feature_size = x_dim*order + 1
        phi_shape = distribution.phi()[0,:].shape

        def lr_func(x, params):
            features = polynomial_features(x)
            output = jnp.einsum('ji,j->i', params['A'], features)
            phi = Jaxtainer.from_array(output, phi_shape)
            return phi

        def polynomial_features(x):
            x_powers = x[:,None]**powers # shape = (x_dim,) 
            x_powers = jnp.array([1., *x_powers.flatten()]) 
            return x_powers

        params = {'A': jax.random.normal(key=prngkey, shape=(feature_size, phi_size))}

        return lr_func, polynomial_features, params

    def initialise(self, x, phi_0=None):
        if phi_0 is None:
            phi_0 = self.distribution.phi(x)
        num_batch = x.shape[0]
        phi_shape = phi_0.shape[1:]
        phi_0 = phi_0.flatten(order='F').reshape(num_batch, -1, order='F')
        features = self.feature_func(x)
        # lstsq returns tuple; only first entry contains least square solution:
        lstsq_coeffs = jnp.linalg.lstsq(features, phi_0)[0]
        self.params = Jaxtainer.from_array(lstsq_coeffs, shapes=self.params.shape)

class NeuralNetwork(AmortisedApproximation):

    _activations = {'relu': jnn.relu, 'sigmoid': jnn.sigmoid}
    _initializers = {'W': jnn.initializers.he_normal(), 'b': jnn.initializers.zeros}

    def __init__(self, distribution, x_dim, prngkey, num_layers=5, width=5, activation='relu', preprocessing=None):
        
        try:
            assert activation.lower() in self._activations
        except (AssertionError, AttributeError):
            raise ValueError(f'Invalid value specified for activation; valid options are: {", ".join(self._activations.keys())}')

        nn_layers, wts = self._create_nn_layers(distribution, x_dim, num_layers, width, activation, prngkey)
        nn = self._create_nn_func(nn_layers, num_layers)
        super().__init__(distribution, phi_func=nn, params=wts, preprocessing=preprocessing)

    def _create_nn_layers(self, distribution, x_dim, num_layers, width, activation, prngkey):
        
        phi_dim = distribution.phi().size

        nn_layers, wts = {}, {}
        for layer in range(num_layers+2):
            if layer == 0: # Input layer
                in_dim, out_dim = x_dim, width
                act_i = self._activations[activation]
            elif layer < num_layers+1:  # Intermediate layer
                in_dim = out_dim = width
                act_i = self._activations[activation]
            else: # Output layer
                in_dim, out_dim = width, phi_dim
                act_i = None
            nn_layers[layer] = self._create_layer_func(act_i) 
            wts[f'W_{layer}'], wts[f'b_{layer}'] = self._initialise_layer_params(prngkey, in_dim, out_dim)
            # Update prngkey so that each layer is initialised differently:
            prngkey = jax.random.split(prngkey, num=1).squeeze()

        # Don't select first batch dimension in phi shape:
        phi_shape = distribution.phi().shape[1:]
        nn_layers[num_layers+2] = self._create_postprocessing_func(phi_shape)

        return nn_layers, wts

    @staticmethod
    def _create_layer_func(activation):
        def layer(x, W, b):
            # NN function will be vectorised over batch dimension of x:
            output = jnp.einsum('ij,i->j', W, x) + b
            if activation is not None:
                output = activation(output)
            return output
        return layer

    def _initialise_layer_params(self, prngkey, in_dim, out_dim):
        W = self._initializers['W'](prngkey, (in_dim, out_dim))
        b = self._initializers['b'](prngkey, (out_dim,))
        return W, b
    
    @staticmethod
    def _create_postprocessing_func(phi_shape):
        def postprocessing_fun(x):
            return Jaxtainer.from_array(x, shapes=phi_shape)
        return postprocessing_fun
    
    @staticmethod
    def _create_nn_func(nn_layers, num_layers):
        def nn(x, params):
            for layer in range(num_layers+2):
                x = nn_layers[layer](x, params[f'W_{layer}'], params[f'b_{layer}'])
            x = nn_layers[num_layers+2](x)
            return x
        return nn

class Preprocessing:

    def __init__(self, preprocess_func):
        self._func = preprocess_func
    
    def __call__(self, x):
        return self._func(x)

    @classmethod
    def std_scaling(cls, data):

        mean = jnp.mean(data, axis=0, keepdims=True)
        std = jnp.std(data, axis=0, keepdims=True)

        def preprocess_func(x):
            return (x - mean)/std
        
        return cls(preprocess_func)

    @classmethod
    def range_scaling(cls, data):

        min_val = jnp.min(data, axis=0, keepdims=True)
        max_val = jnp.max(data, axis=0, keepdims=True)
        range_val = max_val - min_val

        def preprocess_func(x):
            return (x - min_val)/range_val
        
        return cls(preprocess_func)