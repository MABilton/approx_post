import numpy as np
from math import inf

from .algorithms import adam, adagrad
from .bounds import create_bounds_containers, random_container_from_bounds
from ..containers.jax import JaxContainer

EPS = 1e-3

def fit_approximation(loss_and_grad, approx_dist, verbose=False):
    
    # Initialise optimisation parameters:
    optim_params = initialise_optim_params()

    # Create bounds containers:
    bounds_containers = create_bounds_containers(approx_dist.phi_shape, 
                                                 approx_dist.phi_lb, 
                                                 approx_dist.phi_ub)

    # Initialise phi if not specified:
    if approx_dist.phi is None:
        phi_0 = random_container_from_bounds(bounds_containers)
        phi_0 = JaxContainer(phi_0.contents)
    else:
        phi_0 = JaxContainer(approx_dist.phi)    
    
    # Initialise loop variables:
    loop_flag = True
    best_loss = inf
    phi = phi_0

    while loop_flag:
        
        # Compute loss and gradient of metric:
        loss, grad = loss_and_grad(phi)

        # Update parameters with optimisation algorithm:
        phi, optim_params = update_phi(phi, grad, optim_params, bounds_containers)
        
        if verbose:
            print(f'Iteration {optim_params["num_iter"]}:\n   Loss = {loss.item()} \n   Phi = {phi.contents}')
        
        # Store best parameters so far:
        if loss < best_loss:
            best_loss, best_phi = loss, phi

        # Re-check loop condition:
        loop_flag = check_loop_cond(phi, optim_params, bounds_containers)
    
    approx_dist.phi = best_phi

    return approx_dist

def update_phi(phi, grad, optim_params, bounds_containers):

    # Apply optimisation method:
    method = optim_params["method"]
    if method.lower() == "adam":
        update, optim_params = adam(grad, optim_params)
    elif method.lower() == "adagrad":
        update, optim_params = adagrad(grad, optim_params)

    # Update d - convert to np.array in case d is a Jax array:
    phi_new = phi - update
    # Clip updated d so that it lies within param_bounds:
    lb, ub = bounds_containers['lb'], bounds_containers['ub']
    phi_new[phi_new<lb] = lb[phi_new<lb]
    phi_new[phi_new>ub] = ub[phi_new>ub]

    # Update phi_avg:
    optim_params['phi_avg'] = compute_phi_avg(phi_new, optim_params)
    
    return (phi_new, optim_params)

def check_loop_cond(phi, optim_params, bounds_containers, max_iter=1000, phi_threshold=1e-7):
    
    # Check change in phi vs mean:
    phi_avg, num_iter = optim_params['phi_avg'], optim_params['num_iter']
    phi_change_flag = (abs(phi-phi_avg) < phi_threshold).all()

    # Check if phi currently at boundary:
    lb, ub = bounds_containers['lb'], bounds_containers['ub']
    boundary_flag = (abs(phi - lb)<EPS).all() | (abs(phi - ub) < EPS).all()

    # Check all loop conditions:
    if num_iter >= max_iter:
        loop_flag = False
    elif phi_change_flag or boundary_flag:
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
                    'phi_avg': 0.,
                    'num_iter': 0}
    return optim_params

def compute_phi_avg(phi_new, optim_params):
    phi_avg, beta, num_iter = (optim_params[key] for key in ('phi_avg', 'beta_1', 'num_iter'))
    phi_avg = (beta*phi_new + (1-beta)*phi_avg)/(1-beta**num_iter)
    return phi_avg