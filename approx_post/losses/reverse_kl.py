import numpy as np
from math import inf
from arraytainers import Jaxtainer

from .utils import apply_cv, compute_loss_del_w, preprocess_params_and_x

def reverse_kl(approx, joint, method='elbo', use_reparameterisation=True, verbose=False):

    # Create wrapper around forward kl loss function:
    def loss_and_grad(params, x, num_samples):
        params, x = preprocess_params_and_x(params, x)
        phi = Jaxtainer(approx._func_dict['phi'](params, x))
        if method == 'elbo':
            if use_reparameterisation:
                loss, loss_del_phi = elbo_reparameterisation(phi, x, approx, joint, num_samples)
            else:
                loss, loss_del_phi = elbo_controlvariates(phi, x, approx, joint, num_samples)
        elif method == 'selbo':
            if use_reparameterisation:
                loss, loss_del_phi = selbo_reparameterisation(phi, x, approx, joint, num_samples)
            else:
                loss, loss_del_phi = selbo_controlvariates(phi, x, approx, joint, num_samples)
        loss_del_w = compute_loss_del_w(approx, params, x, loss_del_phi)
        # Average over x batches:
        loss, loss_del_w = np.mean(loss, axis=0), np.mean(loss_del_w, axis=0)
        return (loss, loss_del_w)

    return loss_and_grad

def elbo_reparameterisation(phi, x, approx, joint, num_samples):

    # phi.shape = (num_batch, phi_dims)
    # x.shape = (num_batch, num_obs, x_dim)

    # Sample from base distribution then transform samples:
    epsilon_samples = approx._func_dict["sample_base"](num_samples) # epsilon_samples.shape = (num_samples, theta_dim)
    theta_samples = approx._func_dict["transform"](epsilon_samples, phi) # theta_samples.shape = (num_samples, theta_dim)

    # Compute log probabilities and gradient wrt phi:
    approx_lp = approx._func_dict["lp"](theta_samples, phi) # approx_lp.shape = (num_batch, num_samples) 
    approx_del_1 = approx._func_dict["lp_del_1"](theta_samples, phi) # approx_del_1.shape = (num_batch, num_samples, dim_theta) 
    transform_del_phi = approx._func_dict["transform_del_2"](epsilon_samples, phi) 
    # transform_del_phi.shape = (num_batch, num_samples, theta_dim, *phi.shape)
    joint_lp = joint._func_dict["lp"](theta_samples, x) # joint_lp.shape = (num_batch, num_shape)
    joint_del_theta = joint._func_dict["lp_del_1"](theta_samples, x) # joint_del_theta.shape = (num_batch, num_shape, theta_dim)

    # Compute loss and grad:
    loss_samples = joint_lp - approx_lp
    loss = -1*np.mean(loss_samples, axis=1) # loss.shape = (num_batch,)
    grad_samples = np.einsum("abi,abi...->ab...", joint_del_theta, transform_del_phi) \
                 - np.einsum("abi,abi...->ab...", approx_del_1, transform_del_phi)
    grad = -1*np.mean(grad_samples, axis=1) # grad.shape = (num_batch, *phi.shape)

    return (loss, grad)

def elbo_controlvariates(phi, x, approx, joint, num_samples):

    # phi.shape = (num_batch, phi_dims)
    # x.shape = (num_batch, num_obs, x_dim)

    # Draw samples:
    theta_samples = approx._func_dict["sample"](num_samples, phi) # theta_samples.shape = (num_batch, num_samples, theta_dim)

    # Compute log approx._func_dict and gradient wrt phi:
    approx_lp = approx._func_dict["lp"](theta_samples, phi) # approx_lp.shape = (num_batch, num_samples)
    joint_lp = joint._func_dict["lp"](theta_samples, x) # joint_lp.shape = (num_batch, num_samples)
    approx_del_phi = approx._func_dict["lp_del_2"](theta_samples, phi) # approx_del_phi.shape = (num_batch, num_samples, *phi.shape)
    
    # Compute loss and gradient of loss values:
    loss_samples = (joint_lp - approx_lp) # loss_samples.shape = (num_batch, num_samples)
    grad_samples = np.einsum("ab,ab...->ab...", loss_samples, approx_del_phi) # grad_samples.shape = (num_batch, num_samples, *phi.shape)
    # Apply control variates:
    control_variate = approx_del_phi
    num_batch = loss_samples.shape[0]
    loss = -1*apply_cv(loss_samples[:,:,None], control_variate, num_batch, num_samples) # loss.shape = (num_batch,)
    grad = -1*apply_cv(grad_samples, control_variate, num_batch, num_samples) # grad.shape = (num_batch, *phi.shape)
    return (loss, grad)

def selbo_reparameterisation(phi, x, approx, joint, num_samples):

    # phi.shape = (num_batch, phi_dims)
    # x.shape = (num_batch, num_obs, x_dim)

    # Sample from base distribution corresponding to each mixture component then transform samples:
    epsilon_samples = approx._mix_dict["sample_base"](num_samples) 
    # epsilon_samples.shape = (num_mixture, num_samples, theta_dim)
    theta_samples = approx._mix_dict["transform"](epsilon_samples, phi) 
    # theta_samples.shape = (num_mixture, num_samples, theta_dim)

    # Compute log probabilities and gradient wrt phi:
    approx_lp = approx._mix_dict["lp"](theta_samples, phi) # approx_lp.shape = (num_mixture, num_batch, num_samples) 
    approx_del_1 = approx._mix_dict["lp_del_1"](theta_samples, phi) 
    # approx_del_1.shape = (num_mixture, num_batch, num_samples, dim_theta) 
    transform_del_phi = approx._mix_dict["transform_del_2"](epsilon_samples, phi) 
    # transform_del_phi.shape = (num_mixture, num_batch, num_samples, theta_dim, *phi.shape)
    joint_lp = joint._func_dict["lp"](theta_samples, x) # joint_lp.shape = (num_batch, num_samples)
    joint_del_theta = joint._func_dict["lp_del_1"](theta_samples, x) 
    # joint_del_theta.shape = (num_mixture, num_batch, num_shape, theta_dim)

    # Compute loss and grad:
    mix_coeffs = approx._mix_dict['get_coeffs'](phi)
    loss_samples = joint_lp - approx_lp # loss_samples.shape = (num_mixture, num_batch, num_samples) 
    loss_samples = -1*np.mean(loss_samples, axis=2) # loss_samples.shape = (num_mixture, num_batch)
    loss = np.einsum('m,mb->b', mix_coeffs, loss_samples) # loss.shape = (num_batch, )
    grad_samples = np.einsum("abi,mabi...->mab...", joint_del_theta, transform_del_phi) \
                 - np.einsum("mabi,mabi...->mab...", approx_del_1, transform_del_phi)
    # grad_samples.shape = (num_mixture, num_batch, num_samples, *phi.shape)
    grad_samples = -1*np.mean(grad_samples, axis=2) # grad_samples.shape = (num_mixture, num_batch, *phi.shape)
    grad = np.einsum('m,mb...->b...', mix_coeffs, grad_samples) # grad.shape = (num_batch, *phi.shape)

    return (loss, grad)

def selbo_controlvariates(phi, x, approx, joint, num_samples):

    # phi.shape = (num_batch, phi_dims)
    # x.shape = (num_batch, num_obs, x_dim)

    # Draw samples:
    theta_samples = approx._mix_dict["sample"](num_samples, phi) 
    # theta_samples.shape = (num_mixture, num_batch, num_samples, theta_dim)

    # Compute log approx._func_dict and gradient wrt phi:
    approx_lp = approx._mix_dict["lp"](theta_samples, phi) # approx_lp.shape = (num_mixture, num_batch, num_samples)
    joint_lp = joint._mix_dict["lp"](theta_samples, x) # joint_lp.shape = (num_batch, num_samples)
    approx_del_phi = approx._mix_dict["lp_del_2"](theta_samples, phi) 
    # approx_del_phi.shape = (num_mixture, num_batch, num_samples, *phi.shape)
    
    # Compute loss and gradient of loss values:
    loss_samples = (joint_lp - approx_lp) # loss_samples.shape = (num_mixture, num_batch, num_samples)
    coeff_del_phi = approx._mix_dict["coeff_del_phi"]
    grad_samples_term1 = (coeff_del_phi*loss_samples).T 
    # grad_samples_term1.shape = (num_samples, num_batch, num_mixture)
    grad_samples_term2 = np.einsum("mab,mab...->mab...", loss_samples, approx_del_phi).T  
    # grad_samples_term2.shape = (*phi.shape, num_samples, num_batch, num_mixture)
    grad_samples = (grad_samples_term1 + grad_samples_term2).T
    # grad_samples.shape = (*phi.shape, num_samples, num_batch, num_mixture)
    # (grad_samples.T).shape = (num_mixture, num_batch, num_samples, *phi.shape)

    # Reshape samples so that mixture values all along the batch dimension
    loss_samples = loss_samples.reshape(-1, num_samples) # loss_samples.shape = (num_mixture*num_batch, num_samples)
    grad_samples = grad_samples.reshape()

    # Apply control variates:
    control_variate = approx_del_phi
    num_batch = loss_samples.shape[0]
    mix_loss = -1*apply_cv(loss_samples[:,:,None], control_variate, num_batch, num_samples) 
    # mix_loss.shape = (num_mixture*num_batch,)
    mix_grad = -1*apply_cv(grad_samples, control_variate, num_batch, num_samples) 
    # grad.shape = (num_mixture*num_batch, *phi.shape)
    return (loss, grad)