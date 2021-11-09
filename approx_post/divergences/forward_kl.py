import numpy as np
from math import inf

from .cv import apply_cv
from ..optimisation.loop import minimise_loss

def fit(approx_dist, joint_dist=None, initial_samples=None, posterior_samples=None,
        use_reparameterisation=False, verbose=False, num_samples=1000):

    # If we've been given samples to fit 'initial guess' of posterior:
    if initial_samples is not None:
        best_phi, _ = forwardkl_optimloop(approx_dist, joint_dist, initial_samples, 
                                          use_reparameterisation, num_samples)
        approx_dist.phi = best_phi

    # Minimise forward KL divergence:
    best_phi, best_loss = forwardkl_optimloop(approx_dist, joint_dist, posterior_samples, 
                                              use_reparameterisation, verbose, num_samples)

    # Update parameters of approximate dist:
    approx_dist.phi = best_phi

    return approx_dist

def forwardkl_optimloop(approx_dist, joint_dist, provided_samples, use_reparameterisation, verbose, num_samples):
    
    # Create wrapper around forward kl loss function:
    def loss_and_grad(phi):
        
        # If we're given posterior samples, compute forward KL divergence directly:
        if provided_samples is not None:
            loss, grad = forwardkl_sampleposterior(phi, approx_dist, provided_samples)

        # If we're not given posterior samples, we need to use importance sampling:
        else:
            # If we wish to use the reparameterisation trick with importance sampling:
            if use_reparameterisation:
                loss, grad = forwardkl_reparameterisation(phi, approx_dist, joint_dist, num_samples)
            # Otherwise, just use control variates:
            else:
                loss, grad = forwardkl_controlvariates(phi, approx_dist, joint_dist, num_samples)

        return (loss, grad)

    loss_name = 'forward KL divergence'
    best_phi, best_loss = minimise_loss(loss_and_grad, approx_dist, loss_name, verbose)

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
    joint_lp = joint._func_dict["lp"](theta_samples, joint.x)

    # Loss is just cross-entropy (i.e. samples of the joint):
    loss_samples = approx_lp.reshape(-1,1)

    # Call gradient functions:
    approx_del_1 = approx._func_dict["lp_del_1"](theta_samples, phi)
    approx_del_2 = approx._func_dict["lp_del_2"](theta_samples, phi)
    joint_del_1 = joint._func_dict["lp_del_1"](theta_samples, joint.x)
    transform_del_phi = approx._func_dict["transform_del_2"](epsilon_samples, phi)
    joint_del_phi = np.einsum("aj,aj...->a...", joint_del_1, transform_del_phi)
    approx_del_phi = np.einsum("aj,aj...->a...", approx_del_1, transform_del_phi) + approx_del_2
    grad_samples = np.einsum("a,a...->a...", approx_lp, joint_del_phi) + \
                   np.einsum("a,a...->a...", 1-approx_lp, approx_del_phi)

    loss_samples, grad_samples = compute_importance_samples(loss_samples, grad_samples, approx_lp, joint_lp)

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
    joint_lp = joint._func_dict["lp"](theta_samples, joint.x)

    # Loss is just cross-entropy (i.e. samples of the joint):
    loss_samples = approx_lp.reshape(-1,1)

    # Compute gradients:
    approx_del_phi = approx._func_dict["lp_del_2"](theta_samples, phi)
    grad_samples = approx_del_phi

    loss_samples, grad_samples = compute_importance_samples(loss_samples, grad_samples, approx_lp, joint_lp)

    # Apply control variates:
    control_variate = approx_del_phi
    loss = -1*apply_cv(loss_samples, control_variate)
    grad = -1*apply_cv(grad_samples, control_variate)

    return (loss, grad)

def compute_importance_samples(loss_samples, grad_samples, approx_lp, joint_lp):

    log_wts = joint_lp - approx_lp
    max_wts = np.max(log_wts)
    unnorm_wts = np.exp(log_wts-max_wts)
    denom = np.sum(unnorm_wts)

    loss_samples = np.einsum('a,ai->ai', unnorm_wts, loss_samples)/denom
    grad_samples = np.einsum('a,a...->a...', unnorm_wts, grad_samples)/denom

    return (loss_samples, grad_samples)