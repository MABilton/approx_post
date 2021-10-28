from numbers import Number
import numpy as np

PHI_MAX = 1e2
PHI_MIN = -1*PHI_MAX



def create_bound_containers(phi_shape, phi_lb, phi_ub):

    phi_lb = preprocess_bounds(phi_lb, phi_shape, default_val=MIN_PHI)
    phi_lb = preprocess_bounds(phi_lb, phi_shape, default_val=MAX_PHI)

    # Convert bounds to Numpy Containers and place into dictionary:
    phi_bounds = {'lb': NumpyContainer(phi_lb),
                  'ub': NumpyContainer(phi_ub)}

    return phi_bounds

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