from numbers import Number
import numpy as np
from arraytainers import Numpytainer

MAX_PHI = 1e2
MIN_PHI = -1*MAX_PHI

def create_bounds(approx):

    phi_shape, phi_lb, phi_ub = approx.phi_shape, approx.phi_lb, approx.phi_ub

    phi_lb = preprocess_bounds(phi_lb, phi_shape, default_val=MIN_PHI)
    phi_ub = preprocess_bounds(phi_ub, phi_shape, default_val=MAX_PHI)

    # Convert bounds to Numpy Containers and place into dictionary:
    bounds = Numpytainer({'lb':phi_lb, 'ub':phi_ub})

    return bounds

def preprocess_bounds(bounds, phi_shape, default_val):

    if bounds is None:
        try:
            bounds = {key: None for key, _ in phi_shape.items()}
        except AttributeError:
            bounds = [None for _ in phi_shape]

    try:
        bounds_idx = bounds.keys()
    except AttributeError:
        bounds_idx = range(len(bounds))

    for idx in bounds_idx:
        val = bounds[idx]
        if val is None:
            bounds[idx] = default_val*np.ones(phi_shape[idx])
        elif isinstance(val, Number):
            bounds[idx] = val*np.ones(phi_shape[idx])
        else:
            bounds[idx] = val

    return bounds

def random_from_bounds(bounds):
    bounds_range = bounds['ub'] - bounds['lb']
    random_vals = bounds_range.apply(lambda x : np.random.rand(*x.tolist()))
    return random_vals*bounds_range + bounds['lb']