import numpy as np
from .numpy import NumpyContainer

def set_seed(input_seed=42):
    np.random.seed(input_seed)

def random_container_from_shapes(shapes):

    set_seed()

    shape_container = NumpyContainer(shapes)
    random_container = shape_container

    for key_i in shape_container.keys():
        shape_i = shape_container[key_i]
        random_container[key_i] = np.random.random(shape_i)
    
    return random_container