from operator import attrgetter
import jax.numpy as jnp
import numpy as np
from numpy.random import normal as normal_dist
from arraytainers import Numpytainer, Jaxtainer

from .approx import ApproximateDistribution

class AmortisedApproximation(ApproximateDistribution):

    @classmethod
    def nn(cls, approx, x_dim):
        phi, phi_lb, phi_ub = [Numpytainer(x) for x in attrgetter("params", "phi_lb", "phi_ub")(approx)]
        nn, wts = create_nn(x_dim, phi, phi_lb, phi_ub)
        return cls(approx, nn, params=wts)

    def __init__(self, approx, phi_func, params=None):
        self.approx = approx
        self.approx._attr_dict['params'] = params
        self.approx._func_dict['phi'] = phi_func
        self.approx._save_dict['is_amortised'] = True
    
    # See: https://stackoverflow.com/questions/26467564/how-to-copy-all-attributes-of-one-python-object-to-another
    def __getattr__(self, name):
        return getattr(self.approx, name)

def create_nn(x_dim, phi, phi_lb, phi_ub, num_layers=5, width=10, activation='relu'):

    act_fun = relu if activation=='relu' else softmax

    phi_shapes = phi.shape
    phi_dim = phi_shapes.sum().sum_arrays().item()

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
            act_fun = softmax
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
    def nn(wts, x):
        output = np.atleast_2d(x)
        for i in range(num_layers+2):
            W, b = wts[f'W_{i}'], wts[f'b_{i}']
            output = layers[i](output, W, b)
        # Reshape output into correct phi shapes and scale with maximum and minimum values:
        phi = process_nn_output(output, phi_shapes, phi_lb, phi_ub)
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

def process_nn_output(output, phi_shape, phi_lb, phi_ub):
    # Reshape output vector:
    output = Jaxtainer.from_vector(output, phi_shape)
    # Scale values between maximum and minimum values:
    output = (phi_ub - phi_lb)*output + phi_lb
    return output