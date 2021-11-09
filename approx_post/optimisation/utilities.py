import numpy as np
from numpy.random import rand

EPS = 1e-3

def initialise_optim_params():
    optim_params = {'method': "adam", 
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'lr': 10**-1,
                    'eps': 10**-8,
                    'phi_avg': 0.,
                    'num_iter': 0}
    return optim_params

def compute_phi_avg(phi_new, optim_params):
    phi_avg, beta, num_iter = (optim_params[key] for key in ('phi_avg', 'beta_1', 'num_iter'))
    phi_avg = (beta*phi_new + (1-beta)*phi_avg)/(1-beta**num_iter)
    return phi_avg