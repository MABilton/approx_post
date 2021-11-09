import numpy as np
from math import inf

# Internal imports:
from ..optimisation.loop import minimise_loss
from ..optimisation.utilities import apply_cv

def fit(approx_dist, joint_dist, use_reparameterisation=True, verbose=False, num_samples=10):

    # Create wrapper around forward kl loss function:
    def loss_and_grad(phi):
        if use_reparameterisation:
            loss, grad = reversekl_reparameterisation(phi, approx_dist, joint_dist, num_samples)
        else:
            loss, grad = reversekl_controlvariates(phi, approx_dist, joint_dist, num_samples)
        return (loss, grad)

    # Minimise reverse KL divergence:
    loss_name = "reverse KL divergence"
    best_phi, best_loss = minimise_loss(loss_and_grad, approx_dist, loss_name, verbose)

    # Update parameters of approximate dist:
    approx_dist.phi = best_phi

    # Place results in dict:
    results_dict = {'Fitted Distribution': approx_dist,
                    'Loss': best_loss}

    return results_dict

def reversekl_controlvariates(phi, approx, joint, num_samples):

    # Draw samples:
    theta_samples = approx._func_dict["sample"](num_samples, phi)

    # Compute log approx._func_dict and gradient wrt phi:
    approx_lp = approx._func_dict["lp"](theta_samples, phi) 
    joint_lp = joint._func_dict["lp"](theta_samples, joint.x)
    approx_del_phi = approx._func_dict["lp_del_2"](theta_samples, phi)

    # Compute loss and gradient of loss values:
    loss_samples = -1*np.mean(joint_lp - approx_lp, axis=0)
    grad_samples = -1*np.einsum("a,ai->ai", (joint_lp - approx_lp), approx_del_phi)

    # Apply control variates:
    control_variate = approx_del_phi
    loss = apply_cv(loss_samples.reshape(-1,1), control_variate)
    grad = apply_cv(grad_samples, control_variate)

    return (loss, grad)

def reversekl_reparameterisation(phi, approx, joint, num_samples):

    # Sample from base distribution then transform samples:
    epsilon_samples = approx._func_dict["sample_base"](num_samples)
    theta_samples = approx._func_dict["transform"](epsilon_samples, phi)

    # Compute log probabilities and gradient wrt phi:
    approx_lp = approx._func_dict["lp"](theta_samples, phi) 
    approx_del_1 = approx._func_dict["lp_del_1"](theta_samples, phi)
    transform_del_phi = approx._func_dict["transform_del_2"](epsilon_samples, phi)
    joint_lp = joint._func_dict["lp"](theta_samples, joint.x)
    joint_del_theta = joint._func_dict["lp_del_1"](theta_samples, joint.x)

    # Compute loss and grad:
    loss = -1*np.mean(joint_lp - approx_lp, axis=0)
    grad_samples = np.einsum("ai,aij->aj", joint_del_theta, transform_del_phi) \
                   - np.einsum("ai,aij->aj", approx_del_1, transform_del_phi)
    grad = -1*np.mean(grad_samples, axis=0)

    return (loss, grad)