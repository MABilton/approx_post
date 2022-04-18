import copy
import itertools
import inspect
import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
from arraytainers import Jaxtainer

from . import mixture 

class AmortisedApproximation:

    def __init__(self, distribution, phi_func_factory, params_factory, componentwise=True, preprocessing=None, prngkey=None):
        self._distribution = distribution
        self._phi_func_factory = phi_func_factory
        self._params_factory = params_factory
        self._preprocessing = preprocessing
        self._componentwise = componentwise
        self._jaxfunc_dict = self._create_jaxfunc_dict(distribution)
        self._params = self._create_params(distribution, prngkey)

    #
    #   Properties
    #

    @property
    def distribution(self):
        return self._distribution
    
    @property
    def params(self):
        return self._params

    @property
    def is_identical_mixture(self):
        return isinstance(self.distribution, mixture.Identical)

    @property
    def is_componentwise(self):
        return self._componentwise if isinstance(self.distribution, mixture.MixtureApproximation) else False

    #
    #   Function Construction-Related Methods
    #

    def _create_jaxfunc_dict(self, distribution):
        phi_func = self._create_phi_func(distribution)
        logpdf_funcs = self._create_logpdf_funcs(distribution, phi_func)
        jaxfunc_dict = {'phi': phi_func, 
                        'phi_del_x': jax.jacfwd(phi_func, argnums=0),
                        'logpdf_del_x': jax.jacfwd(logpdf_funcs[0], argnums=1),
                        'logpdf_epsilon_del_x': jax.jacfwd(logpdf_funcs[1], argnums=1),
                        'logpdf_del_d': jax.jacfwd(logpdf_funcs[0], argnums=2),
                        'logpdf_epsilon_del_d': jax.jacfwd(logpdf_funcs[1], argnums=2),
                        'phi_del_params': jax.jacfwd(phi_func, argnums=2),}
        for key, func in jaxfunc_dict.items():
            if 'logpdf_epsilon' in key:
                # Vectorise over batch_dim of x and d, and over sample dimension of epsilon:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None,None,None)), in_axes=(None,0,0,None))
            elif 'logpdf' in key:
                # Vectorise over batch_dim of theta, x, and d, and over sample dimension of theta:
                jaxfunc_dict[key] = jax.vmap(jax.vmap(func, in_axes=(0,None,None,None)), in_axes=(0,0,0,None))
            else:
                # Vectorise over batch_dim of x and d:
                jaxfunc_dict[key] = jax.vmap(func, in_axes=(0,0,None))
        return jaxfunc_dict

    def _create_phi_func(self, distribution):
        # Create phi function for each mixture component:
        if self.is_componentwise:
            # If mixture components are identical:
            if self.is_identical_mixture:
                phi_func = self._create_identical_mixture_phi_func(distribution, self._phi_func_factory)
            # If mixture components are different:
            else:
                phi_func = self._create_different_mixture_phi_func(distribution, self._phi_func_factory)
        # Create single function to compute entire phi:
        else:
            phi_func = self._phi_func_factory(distribution.params)
        return phi_func

    @staticmethod
    def _create_identical_mixture_phi_func(distribution, phi_func_factory):

        # NN to predict parameters of each component:
        first_component = list(distribution.components.values())[0]
        noncoeff_func = phi_func_factory(first_component.params)
        vmap_noncoeff_func = jax.vmap(noncoeff_func, in_axes=(None,0))
        # NN to predict mixture coefficients - ensure batch dimension is removed:
        coeffs = jnp.atleast_1d(distribution.coefficients().squeeze())
        coeff_func = phi_func_factory(coeffs)
        # Mixture coefficient keys and non-coefficient keys:
        coeff_key = distribution.coefficient_key
        available_noncoeff_keys = set(distribution.components.keys())

        def phi_func(x, params):
            # Compute phi values of components present in params - can 'freeze' a component by not specifying it in params:
            noncoeff_keys = available_noncoeff_keys.intersection(set(params.keys()))
            noncoeff_params = np.stack([params[key] for key in noncoeff_keys], axis=0)
            noncoeff_phi = vmap_noncoeff_func(x, noncoeff_params)
            phi = Jaxtainer({key: noncoeff_phi[idx,:] for idx, key in enumerate(noncoeff_keys)})
            phi[coeff_key] = coeff_func(x, params[coeff_key])
            return phi

        return phi_func

    @staticmethod
    def _create_different_mixture_phi_func(distribution, phi_func_factory):

        # NNs to predict parameters of each component:
        noncoeff_func_dict = {}
        for key, component in distribution.components.keys():
            noncoeff_nn_dict[key] = phi_func_factory(component.params)
        # NN to predict mixture coefficients:
        coeff_func = phi_func_factory(distribution.coefficients()) 
        # Mixture coefficient keys and non-coefficient keys: 
        coeff_key = distribution.coefficient_key
        available_noncoeff_keys = set(distribution.components.keys())

        def phi_func(x, params):
            noncoeff_keys = available_noncoeff_keys.intersection(set(params.keys()))
            phi = {}
            for key in noncoeff_keys:
                phi[key] = noncoeff_func_dict(x, params[key])
            phi[coeff_key] = coeff_func(x, params[coeff_key])
            return Jaxtainer(phi)

        return phi_func

    @staticmethod
    def _create_logpdf_funcs(distribution, phi_func):
        def logpdf(theta, x, d, params):
            phi = phi_func(x, d, params)
            return distribution.get_function('logpdf')(theta, phi)
        def logpdf_epsilon(epsilon, x, d, params):
            phi = phi_func(x, d, params)
            theta = distribution.get_function('transform')(epsilon, phi)
            return distribution.get_function('logpdf')(theta, phi)
        return logpdf, logpdf_epsilon

    #
    #   Parameter Methods
    #

    def _create_params(self, distribution, prngkey):
        if self.is_componentwise:
            params = {}
            for key, component in distribution.components.items():
                params[key] = self._params_factory(component.params, prngkey)
                prngkey = jax.random.split(prngkey, num=1).squeeze()
            # Ensure that batch dimension is removed from coefficients:
            coeffs = jnp.atleast_1d(distribution.coefficients().squeeze())
            params[distribution.coefficient_key] = self._params_factory(coeffs, prngkey)
        else:
            params = self._params_factory(distribution.params, prngkey)
        return Jaxtainer(params)

    def update(self, new_params):
        self._params = Jaxtainer(new_params)

    #
    #   Phi Methods
    #

    def preprocess_inputs(self, x, d=None):
        x = self._reshape_input(x, ndim=2, num_batch=None)
        d = self._reshape_input(d, ndim=2, num_batch=None)
        if d is None:
            num_batch = x.shape[0]
        else:
            num_batch = max(x.shape[0], d.shape[0]) 
        x = self._reshape_input(x, ndim=2, num_batch=num_batch)
        d = self._reshape_input(d, ndim=2, num_batch=num_batch)
        if self._preprocessing is not None:
            x, d = self._preprocessing(x, d)
        return x, d

    def phi(self, x, d=None):
        x, d = self.preprocess_inputs(x, d)
        return self._jaxfunc_dict['phi'](x, d, self.params)
    
    def phi_del_params(self, x, d=None):
        x, d = self.preprocess_inputs(x, d)
        return self._jaxfunc_dict['phi_del_params'](x, d, self.params)
    
    def phi_del_x(self, x, d=None):
        x, d = self.preprocess_inputs(x, d)
        return self._jaxfunc_dict['phi_del_x'](x, d, self.params)

    def _get_phi(self, phi, x, d):
        if (phi is None) and (x is None):
            raise ValueError('Must specify either phi or x.')
        if (phi is not None) and (not isinstance(phi, Jaxtainer)):
            raise TypeError('Provided phi value is not a Jaxtainer. Did you mean to specify x ' 
                            'instead of phi? If so, explicitly use the "x" keyword argument.')
        if phi is None:
            phi = self.phi(x, d)
        return phi
    
    #
    #   General Distribution Methods
    #

    def logpdf(self, theta, phi=None, x=None, d=None):
        phi = self._get_phi(phi, x, d)
        return self.distribution.logpdf(theta, phi)
    
    def sample(self, num_samples, prngkey, phi=None, x=None, d=None):
        phi = self._get_phi(phi, x, d)
        return self.distribution.sample(num_samples, prngkey, phi=phi)

    def sample_base(self, num_samples, prngkey):
        return self.distribution.sample_base(num_samples, prngkey)

    def transform(self, epsilon, phi=None, x=None, d=None):
        phi = self._get_phi(phi, x, d)
        return self.distribution.transform(epsilon, phi)
    
    def transform_del_2(self, epsilon, phi=None, x=None, d=None):
        phi = self._get_phi(phi, x, d)
        return self.distribution.transform_del_2(epsilon, phi)

    def logpdf_del_1(self, theta, phi=None, x=None, d=None):
        phi = self._get_phi(phi, x, d)
        return self.distribution.logpdf_del_1(theta, phi)
    
    def logpdf_del_2(self, theta, phi=None, x=None, d=None):
        phi = self._get_phi(phi, x, d)
        return self.distribution.logpdf_del_2(theta, phi)

    @staticmethod
    def _reshape_input(val, ndim, num_batch=None):
        if val is not None:
            val = jnp.atleast_1d(val)
            val_original_shape = val.shape
            for _ in range(ndim - val.ndim):
                val = val[None,:]
            if num_batch is not None:
                if (val.shape[0] != num_batch) and (val.shape[0] == 1):
                    val = jnp.broadcast_to(val, shape=(num_batch, *val.shape[1:]))
                elif val.shape[0] != num_batch:
                    num_samples = val.shape[1]
                    raise ValueError('Input theta has unexpected shape. Expected to broadcast theta into shape '
                                    f'(num_batch, num_samples, -1) = ({num_batch}, {num_samples}, -1). '
                                    f'Instead, theta was broadcasted from {val_original_shape} into {val.shape}.')
        return val

    #
    #   Derivatives of Logpdf wrt Obs and Design
    #

    def logpdf_del_x(self, theta, x, d=None):
        x, d = self.preprocess_inputs(x, d)
        theta = self._reshape_input(theta, ndim=3, num_batch=x.shape[0])
        return self._jaxfunc_dict['logpdf_del_x'](theta, x, d, self.params)

    def logpdf_epsilon_del_x(self, epsilon, x, d=None):
        x, d = self.preprocess_inputs(x, d) 
        epsilon = self._reshape_input(epsilon, ndim=2)
        return self._jaxfunc_dict['logpdf_epsilon_del_x'](epsilon, x, d, self.params)

    def logpdf_del_d(self, theta, x, d):
        x, d = self.preprocess_inputs(x, d)
        theta = self._reshape_input(theta, ndim=3, num_batch=x.shape[0])
        return self._jaxfunc_dict['logpdf_del_d'](theta, x, d, self.params)

    def logpdf_epsilon_del_d(self, epsilon, x, d):
        x, d = self.preprocess_inputs(x, d) 
        epsilon = self._reshape_input(epsilon, ndim=2)
        return self._jaxfunc_dict['logpdf_epsilon_del_d'](epsilon, x, d, self.params)

    #
    #   Mixture Distribution Methods
    #

    def _check_if_mixture(self, func_frame):
        if not isinstance(self.distribution, mixture.MixtureApproximation):
            caller_func_name = inspect.getframeinfo(func_frame).function
            raise TypeError('Amortised distribution is not a mixture so it does, '
                            f'not have a {caller_func_name} method.')

    def coefficients(self, phi=None, x=None, d=None):
        self._check_if_mixture(inspect.currentframe())
        phi = self._get_phi(phi, x, d)
        return self.distribution.coefficients(phi)

    def coefficients_del_phi(self, phi=None, x=None, d=None):
        self._check_if_mixture(inspect.currentframe())
        phi = self._get_phi(phi, x, d)
        return self.distribution.coefficients_del_phi(phi)

    def pdf(self, theta, phi=None, x=None, d=None):
        self._check_if_mixture(inspect.currentframe())
        phi = self._get_phi(phi, x, d)
        return self.distribution.pdf(theta, phi)

    def logpdf_epsilon(self, epsilon, phi=None, x=None, d=None):
        self._check_if_mixture(inspect.currentframe())
        phi = self._get_phi(phi, x, d)
        return self.distribution.logpdf_epsilon(epsilon, phi)

    def logpdf_epsilon_del_2(self, epsilon, phi=None, x=None, d=None):
        self._check_if_mixture(inspect.currentframe())
        phi = self._get_phi(phi, x, d)
        return self.distribution.logpdf_epsilon_del_2(epsilon, phi)

    def sample_idx(self, num_samples, prngkey, phi=None, x=None, d=None):
        self._check_if_mixture(inspect.currentframe())
        phi = self._get_phi(phi, x, d)
        return self.distribution.sample_idx(num_samples, prngkey, phi)

    def add_component(self, component=None, prngkey=None):

        if not self.is_componentwise:
            raise TypeError('Cannot add a component to a non-componentwise amortized distribution.')

        # Perform operations on copy initially, in case creation of params fails (e.g. because prngkey not specified)
        new_distribution = copy.deepcopy(self.distribution)

        # Add component to copy of underlying distribution:
        if self.is_identical_mixture:
            if component is not None:
                raise ValueError('Cannot specify component distribution for mixture distribution composed of identical components.')
            new_distribution.add_component()
        else:
            if component is None:
                raise ValueError('Must specify component distribution for mixture distribution composed of different components.')
            new_distribution.add_component(component)
        
        # Create new functions and params:
        new_jaxfunc_dict = self._create_jaxfunc_dict(new_distribution)
        new_params = self._create_params(new_distribution, prngkey)
        for key, val in self.params.items():
            if key != self.distribution.coefficient_key:
                new_params[key] = val
        
        # If no errors raised, can update attributes:
        self._distribution = new_distribution
        self._params = new_params
        self._jaxfunc_dict = new_jaxfunc_dict

class NeuralNetwork(AmortisedApproximation):

    _activations = {'relu': jnn.relu, 'sigmoid': jnn.sigmoid}
    _initializers = {'W': jnn.initializers.he_normal(), 'b': jnn.initializers.zeros}

    def __init__(self, distribution, x_dim, prngkey, d_dim=0, num_layers=5, width=5, activation='relu', componentwise=True, preprocessing=None):
        
        if (not isinstance(activation, str)) and (activation.lower() not in self._activations):
            raise ValueError(f'Invalid value specified for activation; valid options are: {", ".join(self._activations.keys())}')

        nn_func_factory = lambda dist_params: self._nn_func_factory(dist_params, x_dim, d_dim, num_layers, width, activation)
        wts_factory = lambda dist_params, prngkey: self._wts_factory(dist_params, prngkey, x_dim, d_dim, num_layers, width)

        super().__init__(distribution, prngkey=prngkey, phi_func_factory=nn_func_factory, 
                         params_factory=wts_factory, preprocessing=preprocessing, componentwise=componentwise)

    #
    #   NN Function Factory
    #

    def _nn_func_factory(self, params, x_dim, d_dim, num_layers, width, activation):
        
        output_shape = params.shape

        # Create dictionary of functions containing each nn layer:
        nn_layers = {}
        for layer in range(num_layers+2):
            act_i = self._get_ith_layer_activation(layer, activation, num_layers)
            nn_layers[layer] = self._create_ith_layer_func(act_i) 
        nn_layers[num_layers+2] = self._create_postprocessing_func(output_shape)

        # Function which calls nn layers:
        def nn_func(x, d, wts):
            if x.shape[0] != x_dim:
                raise ValueError(f'Unexpected x size. Expected x.size = {x_dim}; instead, x.size = {x.shape[0]}.')
            if d is not None and d.shape[0] != d_dim:
                raise ValueError(f'Unexpected d size. Expected d.size = {d_dim}; instead, d.size = {d.shape[0]}.')
            if d is not None:
                input_i = jnp.concatenate([x, d], axis=0)
            else:
                input_i = x
            for layer in range(num_layers+2):
                input_i = nn_layers[layer](input_i, wts[self.W_key(layer)], wts[self.b_key(layer)])
            phi = nn_layers[num_layers+2](input_i)
            return phi

        return nn_func

    def _get_ith_layer_activation(self, layer, activation, num_layers):
        if layer < num_layers+1:  # Input or Intermediate layer
            act_i = self._activations[activation]
        else: # Output layer
            act_i = None
        return act_i

    @staticmethod
    def _create_ith_layer_func(activation):
        def layer(val, W, b):
            # NN function will be vectorised over batch dimension of x:
            output = jnp.einsum('ij,i->j', W, val) + b
            if activation is not None:
                output = activation(output)
            return output
        return layer

    @staticmethod
    def _create_postprocessing_func(output_shape):
        def postprocessing(x):
            if isinstance(output_shape, Jaxtainer):
                output = Jaxtainer.from_array(x, shapes=output_shape)
            # If NN is predicting mixture coefficients, which is a regular array:
            else:
                output = x.reshape(output_shape)
            return output
        return postprocessing

    #
    #   NN Weights Factory
    #

    def _wts_factory(self, params, prngkey, x_dim, d_dim, num_layers, width):

        if prngkey is None:
            raise ValueError('Must specify a PRNG key to initialise neural network weights.')
        input_dim = x_dim + d_dim
        output_dim = params.size
        nn_wts = {}
        for layer in range(num_layers+2):
            input_dim_i, output_dim_i = self._get_ith_layer_dimensions(layer, input_dim, width, output_dim, num_layers)
            nn_wts[self.W_key(layer)] = self._initializers['W'](prngkey, (input_dim_i, output_dim_i))
            nn_wts[self.b_key(layer)] = self._initializers['b'](prngkey, (output_dim_i,))
            # Update prngkey so that each layer is initialised differently:
            prngkey = jax.random.split(prngkey, num=1).squeeze()
        return nn_wts

    @staticmethod
    def _get_ith_layer_dimensions(layer, input_dim, width, output_dim, num_layers):
        if layer == 0: # Input layer
            in_dim, out_dim = input_dim, width
        elif layer < num_layers+1:  # Intermediate layer
            in_dim = out_dim = width
        else: # Output layer
            in_dim, out_dim = width, output_dim
        return in_dim, out_dim

    @staticmethod
    def W_key(layer):
        return f'W_{layer}'
    
    @staticmethod
    def b_key(layer):
        return f'b_{layer}'

class Preprocessing:

    def __init__(self, preprocess_func, d_specified):
        self._func = preprocess_func
        self._d_specified = d_specified
    
    def __call__(self, x, d=None):
        if self._d_specified and (d is None):
            raise ValueError('d was not provided as an input to a preprocessing function, '
                             "but d values were specified during that function's creation.")
        elif (not self._d_specified) and (d is not None):
            raise ValueError('d was provided as an input to a preprocessing function, '
                             "but d values were not specified during that function's creation.")
        return x, d

    @classmethod
    def std_scaling(cls, x, d=None):

        x_mean = jnp.mean(x, axis=0, keepdims=True)
        x_std = jnp.std(x, axis=0, keepdims=True)
        if d is not None:
            d_mean = jnp.mean(d, axis=0, keepdims=True)
            d_std = jnp.std(d, axis=0, keepdims=True)
            d_specified = True
        else:
            d_specified = False

        def preprocess_func(x, d=None):
            x = (x - x_mean)/x_std
            if d is not None:
                d = (d - d_mean)/d_std
            return x, d
        
        return cls(preprocess_func, d_specified)

    @classmethod
    def range_scaling(cls, x, d=None):
        
        x_min = jnp.min(x, axis=0, keepdims=True)
        x_max = jnp.max(x, axis=0, keepdims=True)
        x_range = x_max - x_min
        if d is not None:
            d_min = jnp.min(d, axis=0, keepdims=True)
            d_max = jnp.max(d, axis=0, keepdims=True)
            d_range = d_max - d_min
            d_specified = True
        else:
            d_specified = False

        def preprocess_func(x, d=None):
            x = (x - x_min)/x_range
            if d is not None:
                d = (d - d_min)/d_range
            return x, d

        return cls(preprocess_func, d_specified)

# class LinearRegression(AmortisedApproximation):

#     def __init__(self, distribution, x_dim, prngkey, order=1, preprocessing=None):
#         lr_func, feature_func, params = self._create_regression_func(distribution, x_dim, order, prngkey)
#         self.feature_func = jax.vmap(feature_func, in_axes=0)
#         super().__init__(distribution, phi_func=lr_func, params=params, preprocessing=preprocessing)

#     def _create_regression_func(self, distribution, x_dim, order, prngkey):

#         # if isinstance(order, int):
#         #     powers = Jaxtainer({key: jnp.arange(0,order) + 1 for key in distribution.phi().keys()}) 
#         # else:
#         #     powers = np.arange(0, Jaxtainer(order)) + 1

#         powers = jnp.arange(0,order) + 1 

#         phi_size = distribution.phi().size 
#         feature_size = x_dim*order + 1
#         phi_shape = distribution.phi()[0,:].shape

#         def lr_func(x, params):
#             features = polynomial_features(x)
#             output = jnp.einsum('ji,j->i', params['A'], features)
#             phi = Jaxtainer.from_array(output, phi_shape)
#             return phi

#         def polynomial_features(x):
#             x_powers = x[:,None]**powers # shape = (x_dim,) 
#             x_powers = jnp.array([1., *x_powers.flatten()]) 
#             return x_powers

#         params = {'A': jax.random.normal(key=prngkey, shape=(feature_size, phi_size))}

#         return lr_func, polynomial_features, params

#     def initialise(self, x, phi_0=None):
#         if phi_0 is None:
#             phi_0 = self.distribution.phi(x)
#         num_batch = x.shape[0]
#         phi_shape = phi_0.shape[1:]
#         phi_0 = phi_0.flatten(order='F').reshape(num_batch, -1, order='F')
#         features = self.feature_func(x)
#         # lstsq returns tuple; only first entry contains least square solution:
#         lstsq_coeffs = jnp.linalg.lstsq(features, phi_0)[0]
#         self.params = Jaxtainer.from_array(lstsq_coeffs, shapes=self.params.shape)