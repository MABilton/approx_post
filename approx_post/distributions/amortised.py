import jax.numpy as jnp
import numpy as np
from numpy.random import normal as normal_dist

# class AmortisedApproximation(ApproximateDistribution):

    # @classmethod
    # def nn(approx, x_values)

    # def __init__(approx, phi_functionm, function_params, x_values):
        
    #     super().__init__(func_dict, attr_dict, save_dict)

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
        output = x
        for i in range(num_layers):
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
        output = jnp.einsum('ij,j->i', W, x) + b
        return act_fun(output)
    
    # Initialise weights using He-Xavier initialisation
    # (see: https://paperswithcode.com/method/he-initialization)
    random_w = normal_dist(loc=0, scale=(2/num_layers)**0.5, size=(out_dim, in_dim))
    wts = {'W': jnp.array(random_w),
           'b': jnp.zeros((out_dim,))}
    
    return (fun, wts)