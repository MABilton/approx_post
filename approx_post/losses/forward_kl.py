import numpy as np
from math import inf
from arraytainers import Jaxtainer

from .utils import apply_cv, compute_loss_del_w, preprocess_params_and_x

def forward_kl(approx, joint=None, provided_samples=None, use_reparameterisation=False):
    
    # Create wrapper around forward kl loss function:
    def loss_and_grad(params, x, num_samples):
        params, x = preprocess_params_and_x(params, x)
        phi = Jaxtainer(approx._func_dict['phi'](params, x))
        # If we're given posterior samples, compute forward KL divergence directly:
        if provided_samples is not None:
            loss, loss_del_phi = forwardkl_sampleposterior(phi, approx, provided_samples)

        # If we're not given posterior samples, we need to use importance sampling:
        else:
            # If we wish to use the reparameterisation trick with importance sampling:
            if use_reparameterisation:
                loss, loss_del_phi = forwardkl_reparameterisation(phi, x, approx, joint, num_samples)
            # Otherwise, just use control variates:
            else:
                loss, loss_del_phi = forwardkl_controlvariates(phi, x, approx, joint, num_samples)

        loss_del_w = compute_loss_del_w(approx, params, x, loss_del_phi)
        loss, loss_del_w = np.mean(loss, axis=0), np.mean(loss_del_w, axis=0)

        return (loss, loss_del_w)

    return loss_and_grad

def forwardkl_sampleposterior(phi, approx, posterior_samples):
    approx_lp = approx._func_dict["lp"](posterior_samples, phi)
    loss = -1*np.mean(approx_lp, axis=1)
    approx_del_phi = approx._func_dict["lp_del_2"](posterior_samples, phi)
    grad = -1*np.mean(approx_del_phi, axis=1)
    return (loss, grad)

def forwardkl_reparameterisation(phi, x, approx, joint, num_samples):

    # phi.shape = (num_batch, phi_dims)
    # x.shape = (num_batch, num_obs, x_dim)

    # Sample from base distribution then transform:
    epsilon_samples = approx._func_dict["sample_base"](num_samples)
    theta_samples = approx._func_dict["transform"](epsilon_samples, phi)
    
    # Evaluate approx lp and likelihood lp at samples:
    approx_lp = approx._func_dict["lp"](theta_samples, phi) # approx_lp.shape = (num_batch, num_samples)
    joint_lp = joint._func_dict["lp"](theta_samples, x) # joint_lp.shape = (num_batch, num_samples)

    # Loss is just cross-entropy (i.e. samples of the joint):
    loss_samples = approx_lp

    # Call gradient functions:
    approx_del_1 = approx._func_dict["lp_del_1"](theta_samples, phi) # approx_del_1.shape = (num_batch, num_samples, theta_dim)
    approx_del_2 = approx._func_dict["lp_del_2"](theta_samples, phi) # approx_del_1.shape = (num_batch, num_samples, *phi.shape)
    joint_del_1 = joint._func_dict["lp_del_1"](theta_samples, x) # approx_del_1.shape = (num_batch, num_samples, theta_dim)
    transform_del_phi = approx._func_dict["transform_del_2"](epsilon_samples, phi) 
    # transform_del_phi.shape = (num_batch, num_samples, theta_dim, *phi.shape)
    joint_del_phi = np.einsum("abj,abj...->ab...", joint_del_1, transform_del_phi) 
    # joint_del_phi.shape = (num_batch, num_samples, *phi.shape)
    approx_del_phi = np.einsum("abj,abj...->ab...", approx_del_1, transform_del_phi) + approx_del_2
    # approx_del_phi.shape = (num_batch, num_samples, *phi.shape)
    
    grad_samples = np.einsum("ab,ab...->ab...", approx_lp, joint_del_phi) + \
                   np.einsum("ab,ab...->ab...", 1-approx_lp, approx_del_phi)
    loss_samples, grad_samples = compute_importance_samples(loss_samples, grad_samples, approx_lp, joint_lp)
    loss = -1*np.mean(loss_samples, axis=1) # loss.shape = (num_batch,)
    grad = -1*np.mean(grad_samples, axis=1) # grad.shape = (num_batch, *phi.shape)
    
    return (loss, grad)

def forwardkl_controlvariates(phi, x, approx, joint, num_samples):

    # phi.shape = (num_batch, phi_dims)
    # x.shape = (num_batch, num_obs, x_dim)

    # Sample from approximating distribution:
    theta_samples = approx._func_dict["sample"](num_samples, phi) # theta_samples.shape = (num_batch, num_samples, theta_dim)

    # Evaluate approx lp and likelihood lp at samples:
    approx_lp = approx._func_dict["lp"](theta_samples, phi)  # approx_lp.shape = (num_batch, num_samples)
    joint_lp = joint._func_dict["lp"](theta_samples, x) # joint_lp.shape = (num_batch, num_samples)

    # Loss is just cross-entropy (i.e. samples of the joint):
    loss_samples = approx_lp

    # Compute gradients:
    approx_del_phi = approx._func_dict["lp_del_2"](theta_samples, phi) # approx_del_phi.shape = (num_batch, num_samples, *phi.shape)
    grad_samples = approx_del_phi

    loss_samples, grad_samples = compute_importance_samples(loss_samples, grad_samples, approx_lp, joint_lp)

    # Apply control variates:
    control_variate = approx_del_phi
    num_batch = loss_samples.shape[0]
    loss = -1*apply_cv(loss_samples, control_variate, num_batch, num_samples) # loss.shape = (num_batch,)
    grad = -1*apply_cv(grad_samples, control_variate, num_batch, num_samples) # grad.shape = (num_batch, *phi.shape)

    return (loss, grad)

def compute_importance_samples(loss_samples, grad_samples, approx_lp, joint_lp):

    log_wts = joint_lp - approx_lp
    max_wts = np.max(log_wts, axis=1).reshape(-1,1)
    unnorm_wts = np.exp(log_wts-max_wts)
    denom = np.sum(unnorm_wts, axis=1).reshape(-1,1)

    loss_samples = np.einsum('ab,ab->ab', unnorm_wts, loss_samples)/denom
    grad_samples = np.einsum('ab,ab...->ab...', unnorm_wts, grad_samples)/denom

    return (loss_samples, grad_samples)