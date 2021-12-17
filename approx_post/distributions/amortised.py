import jax.numpy as jnp
import numpy as np
from numpy.random import normal as normal_dist
from arraytainers import Numpytainer

from .approx import ApproximateDistribution

class AmortisedApproximation(ApproximateDistribution):

    @classmethod
    def nn(cls, approx, x_values):
        phi_shapes = Numpytainer(approx.params).shape
        phi_dim = phi_shapes.sum().sum_arrays().item()
        x_dim = x_values.shape[-1]
        nn, wts = create_nn(x_dim, phi_dim)
        return cls(approx, nn, wts, x_values)

    def __init__(self, approx, phi_func, params, x_values):
        func_dict = approx._func_dict
        super().__init__(approx._func_dict, approx._attr_dict, approx._save_dict)
        self._attr_dict['params'] = params
        self._func_dict['phi'] = phi_func

def create_nn(x_dim, phi_dim, num_layers=5, width=10, activation='relu', output_softmax=True):

    act_fun = relu if activation=='relu' else softmax

    layers, wts = {}, {}
    for i in range(num_layers+2):
        # If creating first layer:
        if i == 0:
            in_dim = x_dim
            out_dim = width
        # If creating last layer:
        elif i==num_layers+1:
            in_dim = width
            out_dim = phi_dim
            act_fun = softmax if output_softmax else act_fun
        # If creating intermediate layer:
        else:
            in_dim = out_dim = width
        # Create layer function and initialise this layer's weights:
        layer_i, wts_i = create_forward_layer(in_dim, out_dim, act_fun, num_layers)        
        # Store initialised layer and weights:
        layers[i] = layer_i
        wts[f'W_{i}'] = wts_i['W']
        wts[f'b_{i}'] = wts_i['b']

    # Define function to call neural network:
    def nn(x, wts):
        if x.ndim < 2:
            x = x.reshape(-1, 1)
        output = x
        for i in range(num_layers+2):
            W, b = wts[f'W_{i}'], wts[f'b_{i}']
            output = layers[i](output, W, b)
        # Reshape output into correct phi shapes:
        # phi = reshape_output(output)
        return output

    return (nn, wts)

def softmax(x):
    exp_x = jnp.exp(x)
    return exp_x/jnp.sum(exp_x)

def relu(x):
    return x*(x>0)

def create_forward_layer(in_dim, out_dim, act_fun, num_layers):
    
    # Define layer function:
    def fun(x, W, b):
        output = jnp.einsum('ij,aj->ai', W, x) + b
        return act_fun(output)
    
    # Initialise weights using He-Xavier initialisation
    # (see: https://paperswithcode.com/method/he-initialization)
    random_w = normal_dist(loc=0, scale=(2/num_layers)**0.5, size=(out_dim, in_dim))
    wts = {'W': jnp.array(random_w),
           'b': jnp.zeros((out_dim,))}
    
    return (fun, wts)