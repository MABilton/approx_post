from numbers import Number
import numpy as np

from ..containers.numpy import NumpyContainer
from ..containers.random import random_container_from_shapes

MAX_PHI = 1e2
MIN_PHI = -1*MAX_PHI

def create_bounds_containers(phi_shape, phi_lb, phi_ub):

    phi_lb = preprocess_bounds(phi_lb, phi_shape, default_val=MIN_PHI)
    phi_ub = preprocess_bounds(phi_ub, phi_shape, default_val=MAX_PHI)

    # Convert bounds to Numpy Containers and place into dictionary:
    bounds_containers = {'lb': NumpyContainer(phi_lb),
                         'ub': NumpyContainer(phi_ub)}

    return bounds_containers

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

def random_container_from_bounds(bound_container):
    lb, ub = bound_container['lb'], bound_container['ub']
    random_vals = random_container_from_shapes(lb.shape())
    random_container = (ub - lb)*random_vals + lb
    return random_container