import numpy as np
from math import inf

# Internal imports:
from ..optimisation.loop import optimisation_loop

def minimise_forwardkl(approx_dist, joint_dist=None, initial_samples=None, posterior_samples=None,
                       use_reparameterisation=False, num_samples=1000):

    # If we've been given samples to fit 'initial guess' of posterior:
    if initial_samples is not None:
        best_phi, _ = forwardkl_optimloop(approx_dist, joint_dist, initial_samples, 
                                          use_reparameterisation, num_samples)
        approx_dist.phi = best_phi

    # Minimise forward KL divergence:
    best_phi, best_loss = forwardkl_optimloop(approx_dist, joint_dist, posterior_samples, 
                                              use_reparameterisation, num_samples)

    # Update parameters of approximate dist:
    approx_dist.phi = best_phi

    # Place results in dict:
    results_dict = {'Fitted Distribution': approx_dist,
                    'Loss': best_loss}

    return results_dict

def forwardkl_optimloop(approx_dist, joint_dist, initial_samples, use_reparameterisation, num_samples):
    
    # Create wrapper around forward kl loss function:
    def loss_and_grad(phi):
        
        # If we're given posterior samples, compute forward KL divergence directly:
        if posterior_samples is not None:
            loss, grad = forwardkl_sampleposterior(phi, approx_dist, posterior_samples)

        # If we're not given posterior samples, we need to use importance sampling:
        else:
            # If we wish to use the reparameterisation trick with importance sampling:
            if use_reparameterisation:
                loss, grad = forwardkl_reparameterisation(phi, approx_dist, joint_dist, num_samples)
            # Otherwise, just use control variates:
            else:
                loss, grad = forwardkl_controlvariates(phi, approx_dist, joint_dist, num_samples)

        return (loss, grad)

    best_phi, best_loss = optimisation_loop(loss_and_grad, approx_dist, loss_name, verbose)

    return (best_phi, best_loss)

def forwardkl_sampleposterior(phi, approx, posterior_samples):
    approx_lp = approx._func_dict["lp"](posterior_samples, phi)
    loss = -1*np.mean(approx_lp, axis=0)
    approx_del_phi = approx._func_dict["lp_del_2"](posterior_samples, phi)
    grad = -1*np.mean(approx_del_phi, axis=0)
    return (loss, grad)

def forwardkl_reparameterisation(phi, approx, joint, num_samples):
    
    # Sample from base distribution then transform:
    epsilon_samples = approx._func_dict["sample_base"](num_samples)
    theta_samples = approx._func_dict["transform"](epsilon_samples, phi)
    
    # Evaluate approx lp and likelihood lp at samples:
    approx_lp = approx._func_dict["lp"](theta_samples, phi)
    joint_lp = joint._func_dict["lp"](theta_samples, x_obs)
    
    # Compute importance weights:
    is_wts = compute_importance_weights(approx_lp, joint_lp, num_samples)

    # Compute loss values:
    loss_samples = is_wts*approx_lp

    # Call gradient functions:
    approx_del_1 = approx._func_dict["lp_del_1"](theta_samples, phi)
    approx_del_2 = approx._func_dict["lp_del_2"](theta_samples, phi)
    joint_del_1 = joint._func_dict["lp_del_1"](theta_samples, x_obs)
    transform_del_phi = approx._func_dict["transform_del_2"](epsilon_samples, phi)
    
    # Use chain rule to compute derivative wrt phi:
    joint_del_phi = np.einsum("aj,aji->ai", joint_del_1, transform_del_phi)
    approx_del_phi = np.einsum("aj,aji->ai", approx_del_1, transform_del_phi) + approx_del_2
    
    # Compute loss grad values:
    grad_samples = np.einsum("a,a,ai->ai", approx_lp, is_wts, joint_del_phi) + \
                   np.einsum("a,a,ai->ai", 1-approx_lp, is_wts, approx_del_phi)

    # Apply control variates:
    control_variate = approx_del_2
    loss = -1*apply_cv(loss_samples, control_variate)
    grad = -1*apply_cv(grad_samples, control_variate)

    return (loss, grad)

def forwardkl_controlvariates(phi, approx, joint, num_samples):

    # Sample from approximating distribution:
    theta_samples = approx._func_dict["sample"](num_samples, phi)

    # Evaluate approx lp and likelihood lp at samples:
    approx_lp = approx._func_dict["lp"](theta_samples, phi)
    joint_lp = joint._func_dict["lp"](theta_samples, x_obs)

    # Compute importance weights:
    is_wts = compute_importance_weights(approx_lp, joint_lp, num_samples)

    # Compute loss values:
    loss_samples = is_wts*approx_lp

    # Compute gradients:
    approx_del_phi = approx._func_dict["lp_del_phi"](theta_samples, phi)
    grad_samples = np.einsum("a,ai->ai", is_wts, approx_del_phi)

    # Define the control variate we'll use:
    cv = approx_del_phi

    # Apply control variates:
    loss = -1*apply_cv(loss_vals, cv)
    grad = -1*apply_cv(grad_samples, cv)

    return (loss, grad)

def compute_importance_weights(approx_lp, joint_lp, num_samples):

    # Compute unnormalised importance weight values:
    unnorm_wts = np.exp(joint_lp - approx_lp)

    # Truncate extreme values (See: 'Truncated Importance Sampling'):
    wts_cutoff = np.mean(unnorm_wts)*num_samples**(1/2) 
    unnorm_wts = np.where(unnorm_wts>wts_cutoff, wts_cutoff, unnorm_wts)

    # Compute normalisation factor:
    normalisation = np.mean(unnorm_wts)

    # Compute normalised weights:
    norm_wts = unnorm_wts/normalisation

    return norm_wts