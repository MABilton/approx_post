import json
from json import JSONDecodeError
import os
import numpy as np
from arraytainers import Jaxtainer
import warnings

from .gaussian import create_gaussian
from .wrappers import wrap_dist_func
from ..optimisation.fit import fit_approximation

class ApproximateDistribution:

    @classmethod
    def gaussian(cls, ndim, params=None, mean_bounds=(None,None), var_bounds=(None,None), cov_bounds=(None,None)):
        func_dict, init_params, phi_lb, phi_ub = create_gaussian(ndim, list(mean_bounds), list(var_bounds), list(cov_bounds))
        params = init_params if params is None else params
        return cls(func_dict, params, ndim, phi_lb=phi_lb, phi_ub=phi_ub)

    def __init__(self, func_dict, params, theta_dim, phi_lb=None, phi_ub=None):
        self._func_dict = preprocess_func_dict(func_dict, theta_dim)
        self.params = Jaxtainer(params)
        self.phi_lb, self.phi_ub = preprocess_bounds(phi_lb, phi_ub, params)
        params_outside_bounds = (self.phi_lb > self.params) | (self.phi_ub < self.params)
        if any(params_outside_bounds):
            warnings.warn('At least one specified phi value lies outside of the valid range of phi values;',
                          'these values will be replaced with a random number between the bounds.')
            self.initialise_phi(mask=params_outside_bounds)

    def initialise_phi(self, seed=1, mask=None):
        np.random.seed(seed)
        rand_val = np.random.rand()
        new_params = (self.phi_ub - self.phi_lb)*rand_val + self.phi_lb
        if mask is not None:
            self.params[mask] = new_params[mask]
        else:
            self.params = new_params

    def fit(self, loss_and_grad, x, num_samples=1000, verbose=False):
        self = fit_approximation(self, loss_and_grad, x, num_samples, verbose)

    def phi(self, x):
        return self._func_dict['phi'](x, self.params)

    def sample(self, num_samples):
        pass
    
    def logpdf(self, theta, x):
        phi = self.phi(self.params, x)
        return self._fun_dict['lp'](theta, phi)

    def logpdf_del_1(self, theta, phi, x):
        pass

    def logpdf_del_2(self, theta, phi, x):
        pass

    def sample_base(self, num_samples):
        pass
    
    def transform(self, epsilon, phi):
        pass
    

    def sample(self, num_samples, x=None): 
        phi = self.phi(self.params, x)
        return self._fun_dict['sample'](num_samples, phi)  

    # def load(self, load_dir):
    #     loaded_params = load_json(load_dir)
    #     loaded_params = Jaxtainer(loaded_params)
    #     try:
    #         assert self.params.keys() == loaded_params.keys()
    #     except AssertionError:
    #         raise ValueError('Loaded parameters are wrong.')

    #     for key, val in self.params.items():
    #         try:
    #             assert val.shape == loaded_params[key].shape
    #         except AssertionError:
    #             raise ValueError('Loaded params are wrong.')

    #     self.params = loaded_params

    # def save(self, save_name='approxdist.json', save_dir='.', indent=4):
    #     to_save = {key: val.tolist() for key, val in self.params.items()}
    #     if save_name[-4:] != 'json':
    #         save_name += ".json"
    #     save_dir = os.path.join(save_dir, save_name)
    #     with open(save_dir, 'w') as f:
    #         json.dump(to_save, f, indent=indent)

def preprocess_func_dict(func_dict, theta_dim):

    try:
        assert all([x in func_dict for x in ('sample', 'lp', 'lp_del_1')])
    except AssertionError:
        error_msg = ("Must include a sample, a lp, and a lp_del_1 method.")
        raise KeyError(error_msg)
    
    missing_reparam_funcs = {x: x in func_dict for x in ('transform', 'sample_base', 'lp_del_2', 'transform_del_2')}
    if not all(missing_reparam_funcs):
        missing_funcs = [func_name for func_name, missing in missing_reparam_funcs.items() if missing]
        warn_msg = (f"Warning: The {', '.join(missing_funcs)} functions have not been supplied.",
                     "The reparameterisation trick cannot be used with this approximating distribution.")
        warnings.warn(" ".join(warn_msg))

    # Create 'default' phi function - just returns param values but repeated along the batch dimension:
    func_dict['phi'] = lambda params, x: np.repeat(params[None,:], x.shape[0], axis=0)

    # Wrap functions to ensure they return correct shapes:
    func_dict = {func_name: wrap_dist_func(func, func_name, theta_dim, dist_type='approx') for func_name, func in func_dict.items()}

    return func_dict

def preprocess_bounds(phi_lb, phi_ub, params, default_min=-1e2, default_max=1e2):
    phi_lb = create_bounds_arraytainer(phi_lb, params, default_min)
    phi_ub = create_bounds_arraytainer(phi_ub, params, default_max)
    if any(phi_ub - phi_lb < 0):
        error_msg = ("At least one phi lower bound value is greater than its respective upper-bound value.")
        raise ValueError(' '.join(error_msg))
    return phi_lb, phi_ub

def create_bounds_arraytainer(bounds, params, default_val):
    ones_arraytainer = np.ones_like(params)
    if bounds is None:
        bounds = default_val*ones_arraytainer
    else:
        bounds = remove_nones_from_bounds(bounds)
        bounds = Jaxtainer(bounds)
        param_keys, bounds_keys = set(params.keys()), set(bounds.keys())
        try:
            assert bounds_keys.issubset(param_keys)
        except AssertionError:
            warn_msg = (f"Specified bounds contains the keys ({' '.join(bounds_keys-param_keys)}), which",
                         "aren't included in the ApproximateDistribution parameters. These keys will",
                         "be ignored.")
            warnings.warn(" ".join(warn_msg))
            bounds = bounds.filter([key for key in param_keys.intersection(bounds_keys)])
        for key in (param_keys - bounds_keys):
            bounds[key] = jnp.array([default_val])
        # Broadcasting will 'fill out' rest of bounds container:
        bounds = bounds*ones_arraytainer
    return bounds

def remove_nones_from_bounds(bounds):
    iterator = bounds.items() if isinstance(bounds, dict) else enumerate(list(bounds))
    for key, val in iterator:
        if bounds[key] is None:
            bounds.pop(key)
        elif isinstance(bounds[key], (list, dict)):
            bounds[idx] = remove_nones_from_bounds(bounds[idx])

def load_json(load_dir):
    try:
        with open(load_dir, 'r') as f:
            dist_json = json.read(f)
    except FileNotFoundError:
        file_dir, file_name
        error_msg = f'File {file_dir} not found in directory {file_dir}.'
        raise FileNotFoundError(error_msg)
    except JSONDecodeError:
        error_msg = (f'Unable to decode contents of {file_name}. Ensure that'
                     f'{file_name} is in a JSON-readable format.')
        raise JSONDecodeError(' '.join(error_msg))
    return dist_json