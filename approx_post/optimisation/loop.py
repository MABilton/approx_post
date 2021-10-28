
import numpy as np

# Internal imports:
from .utilities import initialise_from_bounds

def optimisation_loop(loss_and_grad, phi_shape, phi_0, phi_lb, phi_ub, loss_name, verbose):
    
    # Initialise optimisation parameters:
    optim_params = initialise_optim_params()

    # Create bounds containers:
    phi_bounds = create_phi_bounds(phi_shape, phi_lb, phi_ub)

    # Initialise phi if not specified:
    if phi_0 is None:
        phi_0 = random_container_from_bounds(lb, ub)
   
    # Place the initial parameter guess in a JaxContainer:
    phi_0 = JaxContainer(phi_0)
        
    # Initialise loop variables:
    phi_avg = 0.
    num_iter = 0
    best_loss = inf
    loop_flag = True

    if verbose:
        print(f'Now fitting approximate distribution by minimising {loss_name}.')

    while loop_flag:
        
        # Compute loss and gradient of metric:
        loss, grad = loss_and_grad(phi)

        # Update parameters with optimisation algorithm:
        num_iter += 1
        phi, phi_avg, optim_params = update_phi(grad, optim_params, num_iter, phi, phi_bounds)
        
        if verbose:
            print(f'Iteration {num_iter}: Loss = {loss}, Phi = {phi}')
        
        # Store best parameters so far:
        if loss < best_loss:
            best_loss, best_phi = loss, phi
        
        # Re-check loop condition:
        loop_flag, phi_avg = check_loop_cond(phi, phi_avg, phi_bounds, num_iter)
    
    return (best_phi, best_loss)

def update_phi(grad, optim_params, num_iter, phi, phi_avg, phi_bounds):

    # Apply optimisation method:
    method = optim_params["method"]
    if method.lower() == "adam":
        update, optim_params = adam_update(grad, optim_params, num_iter)
    elif method.lower() == "adagrad":
        update, optim_params = adagrad_update(grad, optim_params)

    # Update d - convert to np.array in case d is a Jax array:
    phi_update = phi - update

    # Clip updated d so that it lies within param_bounds:
    lb, ub = phi_bounds[:,0], phi_bounds[:,1]
    phi_lt_lb, phi_gt_ub = np.array(phi_update < lb), np.array(phi_update > ub)
    phi_update[phi_lt_lb] = lb[phi_lt_lb]
    phi_update[phi_gt_ub] = ub[phi_gt_ub]

    # Update phi_avg:
    phi_avg = compute_avg(phi_update, phi_avg, num_iter)
    return (phi_update, phi_avg, optim_params)

# See: https://www4.stat.ncsu.edu/~lu/ST7901/reading%20materials/Adam_algorithm.pdf
def adam_update(grad, optim_params, num_iter):
    # Unpack parameters:
    beta_1, beta_2, lr, eps = optim_params["beta_1"], optim_params["beta_2"], optim_params["lr"], optim_params["eps"]
    # Unpack velocity and momentum terms from previous iterations:
    m_tm1 = optim_params["m_tm1"] if "m_tm1" in optim_params else 0.
    v_tm1 = optim_params["v_tm1"] if "v_tm1" in optim_params else 0.
    # Compute ADAM update:
    m_t = (1-beta_1)*grad + beta_1*m_tm1
    v_t = (1-beta_2)*grad**2 + beta_2*v_tm1
    tilde_m_t = m_t/(1-beta_1**num_iter)
    tilde_v_t = v_t/(1-beta_2**num_iter)
    update = lr*tilde_m_t/(tilde_v_t**(1/2) + eps)
    # Update m_tm1 and v_tm1 for next iteration:
    optim_params["m_tm1"], optim_params["v_tm1"] = m_t, v_t
    return (update, optim_params)

# See: https://machinelearningjourney.com/index.php/2021/01/05/adagrad-optimizer/
def adagrad_update(grad, optim_params):
    # Unpack parameters:
    lr, eps = optim_params["lr"], optim_params["eps"]
    # Unpack squared gradient:
    s = optim_params["s"] if "s" in optim_params else grad*0
    # Update squared gradient and store for next iteration:
    s += grad**2
    optim_params["s"] = s
    # Perform adagrad update:
    update = lr*grad/((s + eps)**0.5)
    return (update, optim_params)

def initialise_optim_params():
    optim_params = {"method": "adam", 
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "lr": 10**-1,
                    "eps": 10**-8}
    return optim_params

def compute_avg(phi, phi_avg, num_iter, beta=0.9):
    phi_avg = (beta*phi + (1-beta)*phi_avg)/(1-beta**num_iter)
    return phi_avg