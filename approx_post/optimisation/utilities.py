import numpy as np
from np.random import rand

EPS = 1e-3

def check_loop_cond(phi, phi_avg, phi_bounds, num_iter, max_iter=1000, phi_threshold=1e-7):
    
    # Check change in phi vs mean:
    phi_change_flag = np.all(abs(phi - phi_avg) < phi_threshold)

    # Check if phi currently at boundary:
    lb, ub = phi_bounds[:,0], phi_bounds[:,1]
    boundary_flag = np.all((abs(phi - lb) < EPS) | (abs(phi - ub) < EPS))

    # Check if number of iterations exceeds maximum number:
    if num_iter >= max_iter:
        loop_flag = False
    elif phi_change_flag or boundary_flag:
        loop_flag = False
    else:
        loop_flag = True
    return loop_flag

def apply_cv(vals, cv_vals):
    vals = vals.reshape((vals.size,1)) if vals.ndim==1 else vals
    var_cv = np.mean(np.einsum("ai,aj->aij", cv_vals, cv_vals), axis=0)
    cov_vals = np.mean(np.einsum("aj,ai->aij", (vals-np.mean(vals, axis=0)), cv_vals), axis=0)
    a = np.linalg.solve(var_cv, cov_vals)
    val_out = np.mean(vals - np.einsum("ij,ai->aj", a, cv_vals), axis=0)
    return val_out

