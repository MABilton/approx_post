import numpy as np
from math import inf

# Internal imports:
from ..optimisation.loop import optimisation_loop

def minimise_reversekl(approx_dist, joint_dist, use_reparameterisation=True, verbose=False, num_samples=1000):

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
    approx_lp = approxfun_dict["lp"](theta_samples, phi) 
    joint_lp = joint._func_dict["lp"](theta_samples, x_obs)
    approx_del_phi = approx._func_dict["lp_del_phi"](theta_samples, phi)

    # Compute loss and gradient of loss values:
    loss_samples = -1*np.mean(joint_lp - approx_lp, axis=0)
    grad_samples = -1*np.einsum("a,ai->ai", (joint_lp - approx_lp), approx_del_phi)

    # Apply control variates:
    control_variate = approx_del_phi
    loss = apply_cv(loss_samples, control_variate)
    grad = apply_cv(grad_samples, control_variate)

    return (loss, grad)

def reversekl_reparameterisation(phi, approx, joint, num_samples):

    # Sample from base distribution then transform samples:
    epsilon_samples = approx._func_dict["sample_base"](num_samples)
    theta_samples = approx._func_dict["transform"](epsilon_samples, phi)

    # Compute log probabilities and gradient wrt phi:
    approx_lp = approx._func_dict["lp"](theta_samples, phi) 
    approx_del_1 = approx._func_dict["approx_del_theta"](theta_samples, phi)
    transform_del_phi = approx._func_dict["transform_del_phi"](theta_samples, phi)
    joint_lp = joint._func_dict["lp"](theta_samples, x_obs)
    joint_del_theta = joint._func_dict["lp_del_phi"](theta_samples, x_obs)

    # Compute loss and grad:
    loss = -1*np.mean(joint_lp - approx_lp, axis=0)
    grad_samples = np.einsum("ai,ai->a", joint_del_theta, transform_del_phi) \
                   - np.einsum("ai,ai->a", approx_del_1, transform_del_phi)
    grad = -1*np.mean(grad_samples, axis=0)

    return (loss, grad)