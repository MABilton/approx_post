
import numpy as np
import jax.numpy as jnp
from math import inf

class Optimiser:

    def fit(self, approx, loss, x, prngkey, num_samples=None, verbose=False, max_iter=100):
        
        self._initialise_loop_vars()

        params = self._get_params(approx)

        while self._loop_flag:

            loss, loss_del_params = loss.eval(approx, x, prngkey=prngkey, num_samples=num_samples)

            new_params = params - self.step(loss_del_params)
            # Update method will clip params to bounds if necessary:
            approx.update(new_params)

            if verbose:
                self._print_iter(loss)

            if loss < self._best_loss:
                self._best_loss = loss
                best_params = new_params

            self._check_loop_condition(max_iter)
        
        approx.update(best_params)
    
    def _initialise_loop_vars(self):
        self._loop_flag = True
        self._best_loss = inf

    @staticmethod
    def _get_params(approx):

        if hasattr(approx, 'params'):
            params = approx.params
        else:
            params = approx._phi

        return params

    def _print_iter(self, num_iter):
        pass

    def _check_loop_condition(self, max_iter):
        if self.num_iter >= max_iter:
            self._loop_flag = False

class Adam(Optimiser):

    def __init__(self, beta_1=0.9, beta_2=0.999, lr=1e-1, eps=1e-8):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lr = lr
        self.eps = eps
        
    def step(self, grad):

        m_t = self._compute_exp_avg(new_val=grad, current_avg=self._m_tm1, wt=self.beta_1)
        v_t = self._compute_exp_avg(new_val=grad**2, current_avg=self._v_tm1, wt=self.beta_2)

        update = self.lr*m_t/(v_t**0.5 + eps)

        self._update_optim_params(m_t, v_t)

        return update

    def _initialise_optim_params(self):
        self.m_tm1 = 0
        self.v_tm1 = 0
        self.num_iter = 0

    def _compute_exp_avg(self, new_val, current_avg, wt):
        new_avg = (1-wt)*current_avg + wt*new_val
        return self._apply_bias_correction(new_avg, wt)

    def _apply_bias_correction(self, new_avg, wt):
        return new_avg/(1-wt**self.num_iter)

    def _update_optim_params(self):
        self.m_tm1 = m_t
        self.v_tm1 = v_t
        self.num_iter += 1

class AdaGrad(Optimiser):

    def __init__(self, lr=1e-1, eps=1e-8):
        self.lr = lr
        self.eps = eps
    
    def step(self, grad):
        s = self._s + grad**2
        update = self.lr*grad/((s + self.eps)**0.5)
        self._update_optim_params(s)
        return update

    def _initialise_optim_params(self):
        self._s = 0
        self.num_iter = 0

    def _update_optim_params(self, s):
        self._s += s
        self.num_iter += 1