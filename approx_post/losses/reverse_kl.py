import numpy as np
from math import inf

# Internal imports:
from .utils import apply_cv

def reverse_kl(approx, joint, use_reparameterisation=True, verbose=False, num_samples=100):

    # Create wrapper around forward kl loss function:
    def loss_and_grad(params, x):
        phi = approx._func_dict['phi'](params, x)
        if use_reparameterisation:
            loss, loss_del_phi = reversekl_reparameterisation(phi, x, approx, joint, num_samples)
        else:
            loss, loss_del_phi = reversekl_controlvariates(phi, x, approx, joint, num_samples)
        phi_del_w = approx._func_dict['phi_del_w'](params, x)
        loss_del_w = np.einsum('ai,ij->aj', phi_del_w, loss_del_phi)
        # Average over x batches:
        loss, loss_del_w = np.mean(loss, axis=0), np.mean(loss_del_w, axis=0)
        return (loss, loss_del_w)

    return loss_and_grad

def reversekl_reparameterisation(phi, x, approx, joint, num_samples):

    # Sample from base distribution then transform samples:
    epsilon_samples = approx._func_dict["sample_base"](num_samples)
    theta_samples = approx._func_dict["transform"](epsilon_samples, phi)

    # Compute log probabilities and gradient wrt phi:
    approx_lp = approx._func_dict["lp"](theta_samples, phi) 
    approx_del_1 = approx._func_dict["lp_del_1"](theta_samples, phi)
    transform_del_phi = approx._func_dict["transform_del_2"](epsilon_samples, phi)
    joint_lp = joint._func_dict["lp"](theta_samples, x)
    joint_del_theta = joint._func_dict["lp_del_1"](theta_samples, x)

    # Compute loss and grad:
    loss = -1*np.mean(joint_lp - approx_lp, axis=1)
    grad_samples = np.einsum("abi,bi...->ab...", joint_del_theta, transform_del_phi) \
                 - np.einsum("bi,bi...->b...", approx_del_1, transform_del_phi)
    grad = -1*np.mean(grad_samples, axis=1)

    return (loss, grad)

def reversekl_controlvariates(phi, x, approx, joint, num_samples):

    # Draw samples:
    theta_samples = approx._func_dict["sample"](num_samples, phi)

    # Compute log approx._func_dict and gradient wrt phi:
    approx_lp = approx._func_dict["lp"](theta_samples, phi) 
    joint_lp = joint._func_dict["lp"](theta_samples, x)
    approx_del_phi = approx._func_dict["lp_del_2"](theta_samples, phi)

    # Compute loss and gradient of loss values:
    loss_samples = -1*(joint_lp - approx_lp)
    grad_samples = -1*np.einsum("ab,b...->ab...", loss_samples, approx_del_phi)
    print(loss_samples.reshape(-1,num_samples,1).shape)
    # Apply control variates:
    control_variate = approx_del_phi
    loss = apply_cv(loss_samples.reshape(-1,num_samples,1), control_variate)
    grad = apply_cv(grad_samples, control_variate)

    return (loss, grad)