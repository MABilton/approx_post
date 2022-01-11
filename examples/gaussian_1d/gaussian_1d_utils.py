import json
import numpy as np
from numpy.random import multivariate_normal as mvn
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as jmvn
from math import inf
from itertools import product
from tqdm import tqdm
from scipy.optimize import minimize
from scipy import integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
mpl.rcParams['figure.dpi'] = 100

def load_json(to_load):
    with open(to_load, 'r') as f:
        loaded = json.load(f)
    return loaded

def save_json(to_save, save_name):
    with open(save_name, 'w') as f:
        json.dump(to_save, f, indent=4)

def create_data(model, true_theta, noise_std, num_samples):
    mean = model(true_theta)
    samples = mvn(mean, noise_std, num_samples)
    return samples.reshape(num_samples, -1)

def create_gaussian(mean, std):
    def gaussian(theta):
        return jmvn.logpdf(theta, mean, std)
    return gaussian

def create_laplace_approx(model, x_obs, noise_std, prior_mean, prior_std, x0=1):
    
    def loss(theta):
        likelihood_term = ((x_obs-model(theta))/noise_std)**2
        likelihood_term = jnp.sum(likelihood_term, axis=0)
        prior_term = ((theta-prior_mean)/prior_std)**2
        return (likelihood_term + prior_term)[0]

    loss_grad = jax.jacfwd(loss) 
    result = minimize(loss, x0=x0, jac=lambda theta: loss_grad(theta)[0])
    theta_map = result.x.reshape(1,)

    model_grad = jax.jacfwd(model)
    laplace_std = ((model_grad(theta_map)/noise_std)**2 + (1/prior_std)**2)**-1
    laplace_std = laplace_std.reshape(1,1)

    def laplace_approx(theta):
        return jmvn.logpdf(theta.reshape(-1,1), theta_map, laplace_std)

    print(f'Laplace Approximation: Mean = {theta_map[0]}, Standard deviation = {laplace_std[0]}')

    return laplace_approx

def create_posterior(model, x_obs, noise_std, prior_mean, prior_std):
    jmvn_vmap = jax.vmap(jmvn.logpdf, in_axes=(None,0,None), out_axes=0)
    def unnorm_posterior_lp(theta):
        if not isinstance(theta, jnp.DeviceArray):
            theta = jnp.array(theta).reshape(-1,1)
        x_pred = model(theta)
        return np.sum(jmvn_vmap(x_obs, x_pred, noise_std), axis=1) + jmvn.logpdf(theta, prior_mean, prior_std)
    unnorm_posterior = lambda theta : np.exp(unnorm_posterior_lp(theta))
    norm_const = integration_scheme(unnorm_posterior)
    log_norm_const = np.log(norm_const)
    posterior_lp = lambda theta: unnorm_posterior_lp(theta) - log_norm_const
    return posterior_lp

def compute_loss(mean_vals, std_vals, approx, joint=None, posterior=None, num_samples=1000, divergence='forward'):
    
    if posterior is not None:
        use_posterior = True
    elif joint is not None:
        use_posterior = False
    else:
        raise ValueError('Must specify either posterior or joint function.')

    loss_vals = {key: [] for key in ('loss', 'mean', 'std')}
    for mean_i, std_i in tqdm(product(mean_vals, std_vals)):
        loss_vals['mean'].append(mean_i), loss_vals['std'].append(std_i)
        phi_i = {'mean': np.array([mean_i]), 'chol_diag': np.array([std_i]), 'chol_lowerdiag': np.array([])}
        if use_posterior:
            loss_i = compute_loss_with_posterior(phi_i, approx, posterior, divergence)
        else:
            loss_i = compute_loss_without_posterior(phi_i, approx, joint, num_samples, divergence)
        # Convert to number value:
        try:
            loss_i = loss_i.item()
        except AttributeError:
            pass
        loss_vals['loss'].append(loss_i)

    return loss_vals

def compute_loss_with_posterior(phi, approx, posterior_lp, divergence):
    approx_lp = lambda theta : approx._func_dict["lp"](theta, phi)
    def kl_func(theta):
        log_approx = approx_lp(theta)
        log_posterior = posterior_lp(theta)
        if divergence == 'forward':
            return np.exp(log_posterior)*(log_posterior - log_approx)
        elif divergence == 'reverse':
            return np.exp(log_approx)*(log_approx - log_posterior)
    kl_loss = integration_scheme(kl_func) #integrate.quadrature(kl_func, -inf, inf)
    return kl_loss

def compute_loss_without_posterior(phi, approx, joint, num_samples, divergence):

    theta_samples = approx._func_dict["sample"](num_samples, phi)
    approx_lp = approx._func_dict["lp"](theta_samples, phi)
    joint_lp = joint._func_dict["lp"](theta_samples, joint.x)

    if divergence == 'forward':
        loss_samples = approx_lp
        log_wts = joint_lp - approx_lp
        max_wts = np.max(log_wts)
        unnorm_wts = np.exp(log_wts-max_wts)
        loss = -1*np.sum(unnorm_wts*loss_samples)/np.sum(unnorm_wts)

    elif divergence == 'reverse':
        loss_samples = joint_lp - approx_lp
        loss = -1*np.mean(loss_samples)

    return loss

def plot_loss(loss_vals, mean_lims=None, var_lims=None, loss_lims=None, num_levels=100, num_ticks=10, plot_pts=False):
    
    plot_vals = {key: np.array(val) for key, val in loss_vals.items()}
    
    grid_shape = []
    for key in ('std', 'mean'):
        _, num_unique = np.unique(plot_vals[key], return_counts=True)
        grid_shape.append(num_unique[0])

    loss_lims = [np.nanmin(plot_vals['loss']), np.nanmax(plot_vals['loss'])] if loss_lims is None else list(loss_lims)
    for idx, val in enumerate(loss_lims):
        if val is None:
            new_val = np.nanmin(plot_vals['loss']) if idx==0 else np.nanmax(plot_vals['loss'])
            loss_lims[idx] = new_val
    plot_vals['loss'] = np.clip(plot_vals['loss'], *loss_lims)
    levels = np.linspace(*loss_lims, num_levels)
    
    # Create surface plot:
    fig, ax = plt.subplots()
    contour_fig = ax.contourf(*(plot_vals[key].reshape(grid_shape) for key in ['mean', 'std', 'loss']), levels=levels, cmap=cm.coolwarm)
    ticks = np.linspace(*loss_lims, num_ticks)
    cbar = fig.colorbar(contour_fig, ticks=ticks)
    cbar.set_ticklabels([f'{x:.1f}' for x in ticks])
    cbar.set_label('Loss', rotation=270, labelpad=15)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Standard Deviation')
    ax.set_xlim(mean_lims)
    ax.set_ylim(var_lims)

    # Plot data points:
    if plot_pts:
        plt.plot(*(plot_vals[key] for key in ['mean', 'std']), 'x', color='black', markersize=3)

    # Plot global minimum:
    min_idx = np.argmin(plot_vals['loss']) 
    plt.plot(*(plot_vals[key].flatten()[min_idx] for key in ['mean', 'std']), 'x', color='yellow', markersize=6)
    fig.patch.set_facecolor('white')
    plt.show()

def plot_distributions(*lp_funcs, theta_lims=(-5,5), num_pts=1000):
    theta = np.linspace(*theta_lims, num_pts)
    fig, ax = plt.subplots()
    for lp in lp_funcs:
        pdf_vals = np.exp(lp(theta))
        ax.plot(theta, pdf_vals)
        ax.fill_between(theta, pdf_vals, alpha=0.3)
    fig.patch.set_facecolor('white')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Probability Density')
    plt.show()

def integration_scheme(func, theta_lims=(-10,10), num_steps=2**13+1):
    theta = np.linspace(*theta_lims, num_steps).reshape(-1,1)
    dx = abs(theta[1] - theta[0])
    func_samples = func(theta)
    integral_val = integrate.romb(func_samples, dx=dx)
    return integral_val

def set_integration_limits(theta_min, theta_max, steps=None):
    steps = THETA_LIMS[-1] if steps is None else steps