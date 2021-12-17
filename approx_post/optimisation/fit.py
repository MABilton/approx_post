import numpy as np
from math import inf
from arraytainers import Jaxtainer
from .algorithms import adam, adagrad
from .bounds import create_bounds

from ..distributions.amortised import AmortisedApproximation

EPS = 1e-3

def fit_approximation(loss_and_grad, approx_dist, x, verbose=False):
    
    # Ensure x contains a 'batch' and 'num_obs' dimension:
    for _ in range(3 - x.ndim):
        x = x[None,:] 

    # Initialise optimisation parameters:
    optim_params = initialise_optim_params()

    # Create bounds containers:
    bounds_containers = create_bounds(approx_dist)
    
    # Initialise loop variables:
    loop_flag = True
    best_loss = inf 
    params = Jaxtainer(approx_dist.params)
    clip_output = not isinstance(approx_dist, AmortisedApproximation)
    
    while loop_flag:
        
        # Compute loss and gradient of metric:
        loss, grad = loss_and_grad(params, x)

        # Update parameters with optimisation algorithm:
        params, optim_params = update_params(params, grad, optim_params, bounds_containers, clip_output)
        
        if verbose:
            print(f'Iteration {optim_params["num_iter"]}:\n   Loss = {loss.item()}, Params = {params.unpacked}')
        
        # Store best parameters so far:
        if loss < best_loss:
            best_loss, best_params = loss, params

        # Re-check loop condition:
        loop_flag = check_loop_cond(params, optim_params)
    
    approx_dist._attr_dict['params'] = best_params.unpacked

    return approx_dist

def update_params(params, grad, optim_params, bounds_containers, clip_output):

    # Apply optimisation method:
    method = optim_params["method"]
    if method.lower() == "adam":
        update, optim_params = adam(grad, optim_params)
    elif method.lower() == "adagrad":
        update, optim_params = adagrad(grad, optim_params)

    # Update d - convert to np.array in case d is a Jax array:
    params_new = params - update

    # Clip updated d so that it lies within param_bounds:
    if clip_output:
        lb, ub = bounds_containers['lb'], bounds_containers['ub']
        params_new[params_new<lb] = lb[params_new<lb]
        params_new[params_new>ub] = ub[params_new>ub]

    # Update phi_avg:
    optim_params['params_avg'] = compute_params_avg(params_new, optim_params)
    
    return (params_new, optim_params)

def check_loop_cond(params, optim_params, max_iter=1000, params_threshold=1e-7):
    
    # Check change in phi vs mean:
    params_avg, num_iter = optim_params['params_avg'], optim_params['num_iter']
    params_change_flag = (abs(params-params_avg) < params_threshold).all()

    # Check all loop conditions:
    if (num_iter >= max_iter) or params_change_flag:
        loop_flag = False
    else:
        loop_flag = True

    return loop_flag

def initialise_optim_params():
    optim_params = {'method': "adam", 
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'lr': 10**-1,
                    'eps': 10**-8,
                    'params_avg': 0.,
                    'num_iter': 0}
    return optim_params

def compute_params_avg(params_new, optim_params):
    params_avg, beta, num_iter = (optim_params[key] for key in ('params_avg', 'beta_1', 'num_iter'))
    params_avg = (beta*params_new + (1-beta)*params_avg)/(1-beta**num_iter)
    return params_avg