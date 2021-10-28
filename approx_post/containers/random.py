import numpy as np

def set_seed(input_seed=42):
    np.random.seed(input_seed)

def random_container_from_shapes(shapes):

    set_seed()

    random_container = NumpyContainer(shapes)

    for key_i in shape_container.keys:
        shape_i = random_container[key_i]
        random_container[key_i] = np.random.random(shape_i)
    
    return random_container

def random_container_from_bounds(lb, ub):
    random_vals = random_container_from_shapes(lb.shape)
    random_container = (ub - lb)*random_vals + lb
    return random_container