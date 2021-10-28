import numpy as np
from math import inf

# Internal imports:
from ..optimisation.loop import optimisation_loop

def minimise_reversekl(approxfun_dict, jointfun_dict, x_obs, phi_shape, phi_0=None, phi_lb=None, 
                       phi_ub=None, use_reparameterisation=True, verbose=False, num_samples=1000):

    # Create wrapper around forward kl loss function:
    def loss_and_grad(phi):
        if use_reparameterisation:
            loss, grad = reversekl_reparameterisation(phi, approxfun_dict, jointfun_dict, x_obs)
        else:
            loss, grad = reversekl_controlvariates(phi, approxfun_dict, jointfun_dict, x_obs)
        return (loss, grad)

    # Minimise reverse KL divergence:
    loss_name = "reverse KL divergence"
    best_phi, best_loss = minimise_loss(loss_and_grad, phi_shape, phi_0, phi_lb, 
                                        phi_ub, loss_name, verbose)
    return (loss, grad)

def reversekl_controlvariates(phi, approxfun_dict, jointfun_dict, x_obs):

    # Draw samples:
    theta_samples = approxfun_dict["sample"](num_samples, phi)

    # Compute log probabilities and gradient wrt phi:
    approx_lp = approxfun_dict["lp"](theta_samples, phi) 
    joint_lp = jointfun_dict["lp"](theta_samples, x_obs)
    approx_del_phi = approxfun_dict["lp_del_phi"](theta_samples, phi)

    # Compute loss and gradient of loss values:
    loss_samples = -1*np.mean(joint_lp - approx_lp, axis=0)
    grad_samples = -1*np.einsum("a,ai->ai", (joint_lp - approx_lp), approx_del_phi)

    # Apply control variates:
    control_variate = approx_del_phi
    loss = apply_cv(loss_samples, control_variate)
    grad = apply_cv(grad_samples, control_variate)

    return (loss, grad)

def reversekl_reparameterisation(phi, approxfun_dict, jointfun_dict, x_obs):

    # Sample from base distribution then transform samples:
    epsilon_samples = approxfun_dict["sample"](num_samples)
    theta_samples = approxfun_dict["transform"](epsilon_samples, phi)

    # Compute log probabilities and gradient wrt phi:
    approx_lp = approxfun_dict["lp"](theta_samples, phi) 
    approx_del_1 = approxfun_dict["approx_del_theta"](theta_samples, phi)
    transform_del_phi = approxfun_dict["transform_del_phi"](theta_samples, phi)
    joint_lp = jointfun_dict["lp"](theta_samples, x_obs)
    joint_del_theta = jointfun_dict["lp_del_phi"](theta_samples, x_obs)

    # Compute loss and grad:
    loss = -1*np.mean(joint_lp - approx_lp, axis=0)
    grad_samples = np.einsum("ai,ai->a", joint_del_theta, transform_del_phi) \
                   - np.einsum("ai,ai->a", approx_del_1, transform_del_phi)
    grad = -1*np.mean(grad_samples, axis=0)

    return (loss, grad)