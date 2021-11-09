import numpy as np
from functools import reduce
from more_itertools.more import always_iterable

from ..containers.array import ArrayContainer
from ..containers.numpy import NumpyContainer
    
def apply_cv(val, cv):

    val_in = NumpyContainer([val]) if not issubclass(type(val), ArrayContainer) else val        

    val_vec = vectorise_values(val_in)
    cv_vec = vectorise_values(cv)
    var = np.mean(np.einsum("ai,aj->aij", cv_vec, cv_vec), axis=0)
    cov = np.mean(np.einsum("aj,ai->aij", (val_vec-np.mean(val_vec, axis=0)), cv_vec), axis=0)
    a = np.linalg.solve(var, cov)

    val_vec = np.mean(val_vec - np.einsum("ij,ai->aj", a, cv_vec), axis=0)
    val = reshape_output(val_vec, val)
    return val

def vectorise_values(val):
    shapes = val.shape
    shapes = NumpyContainer(shapes)
    num_samples = shapes[0][0]
    val_vec = np.reshape(val, (num_samples,-1))
    val_vec = np.concatenate(val_vec.values(), axis=1)
    return val_vec

def reshape_output(val_vec, val):
    out_shapes = val.shape[1:]
    if issubclass(type(val), ArrayContainer):
        val_out = val.copy()
        i = 0
        for key, shape in out_shapes.items():
            num_elem = reduce(lambda x,y: x*y, shape)
            val_i = val_vec[i:i+num_elem]
            i += num_elem
            val_out[key] = val_i.reshape(shape)
    else:
        val_out = val_vec.reshape(out_shapes)
    return val_out