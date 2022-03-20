import numpy as np
import jax.numpy as jnp
from arraytainers import Jaxtainer

class Loss:

    #
    #   Chain rule to compute loss del params
    #

    def _compute_loss_del_params(self, loss_del_phi, x, approxdist):

        # loss_del_phi.shape = (num_batch, phi_shape)
        # x.shape = (num_batch, x_dim)

        if not hasattr(approxdist, 'phi_del_params'):
            loss_del_params = loss_del_phi

        else:
            num_batch = x.shape[0]
            phi_size = loss_del_phi[0,:].size
            phi_sizes_arraytainer = loss_del_phi[0,:].sizes
            params_shape = approxdist.params.shape

            loss_del_phi = loss_del_phi.flatten(order='F').reshape(num_batch, phi_size, order='F')
            # shape = (num_batch, phi_dim)

            phi_del_params = approxdist.phi_del_params(x)
            phi_del_params = self._vectorise_phi_del_params(phi_del_params, num_batch, phi_sizes_arraytainer, params_shape)
            # shape = (num_batch, phi_dim, param_dim)

            loss_del_params = jnp.einsum('ai,aij->aj', loss_del_phi, phi_del_params) 
            # shape = (num_batch, param_ndim)
            loss_del_params = Jaxtainer.from_array(loss_del_params, shapes=(num_batch, params_shape), order='F')

        return loss_del_params

    @staticmethod
    def _vectorise_phi_del_params(phi_del_params, num_batch, phi_sizes_arraytainer, params_shape):
        
        phi_del_params = phi_del_params.reshape(num_batch, phi_sizes_arraytainer, -1, order='F')

        param_key_order = tuple(params_shape.keys())
        for key, inner_arraytainer in phi_del_params.items():
            array_list = [inner_arraytainer[param_key] for param_key in param_key_order]
            phi_del_params[key] = jnp.concatenate(array_list, axis=2)
        
        return jnp.concatenate(phi_del_params.list_elements(), axis=1)

    @staticmethod
    def _avg_over_batch_dim(loss, loss_del_phi):
        # Use np.mean on loss_del_phi since it's an arraytainer:
        return jnp.mean(loss, axis=0), np.mean(loss_del_phi, axis=0)

    #
    #   Control variates
    #

    def _apply_controlvariates(self, val, cv):

        num_batch, num_samples = self._get_batch_and_sample_size_from_cv(cv)

        val_vec = self._vectorise_controlvariate_input(val, new_shape=(num_batch, num_samples, -1))
        cv_vec = self._vectorise_controlvariate_input(cv, new_shape=(num_batch, num_samples, -1))
        
        var = self._compute_covariance(cv_vec, cv_vec) # shape = (num_batches, num_samples, dim_cv, dim_cv)
        cov = self._compute_covariance(cv_vec, val_vec, subtract_arg_2_mean=True) # shape = (num_batches, num_samples, dim_cv, dim_val)

        cv_samples = self._compute_controlvariate_samples(val_vec, cv_vec, cov, var)
        val = self._reshape_controlvariate_output(cv_samples, val)

        return val
    
    @staticmethod
    def _get_batch_and_sample_size_from_cv(cv):
        first_array = cv.list_elements()[0]
        num_batch, num_sample = first_array.shape[0:2]
        return num_batch, num_sample

    @staticmethod
    def _vectorise_controlvariate_input(val, new_shape):
        # Need order = 'F' here for correct ordering of axes; 
        # Note that flatten method shapes arraytainer into a single 1D array
        return val.flatten(order='F').reshape(new_shape, order='F')

    @staticmethod
    def _compute_covariance(val_1, val_2, subtract_arg_2_mean=False):
        if subtract_arg_2_mean:
            val_2 = val_2 - np.mean(val_2, axis=1, keepdims=True)
        return np.mean(np.einsum("abi,abj->abij", val_1, val_2), axis=1)
    
    def _compute_controlvariate_samples(self, val_vec, cv_vec, cov, var):
        # Can have singular matrix when dealing with mixture distributions - need to add jitter:
        a = -1*self._solve_matrix_system(var, cov) # shape = (num_batch, num_samples, dim_cv, dim_val)
        return val_vec + np.einsum("aij,abi->abj", a, cv_vec) # shape = (num_batch, num_samples, num_val)

    @staticmethod
    def _solve_matrix_system(A, b, start_jitter=1e-8, max_jitter=1e-5, delta_jitter=1e1):
        
        is_solved = False
        jitter = 0 
        while not is_solved:

            try:
                A_jitter = A + jitter*jnp.identity(A.shape[0])
                x = -1*np.linalg.solve(A_jitter, b) 
                # x will contain inf or nan entries if A is singular:
                if jnp.any(jnp.isnan(x) | jnp.isinf(x)):
                    raise np.linalg.LinAlgError
                is_solved = True

            except np.linalg.LinAlgError as e:
                if jitter >= max_jitter:
                    raise e
                elif jitter < start_jitter:
                    jitter = start_jitter
                else:
                    jitter *= delta_jitter

        return x

    @staticmethod
    def _reshape_controlvariate_output(cv_samples, val):
        # order='F' since val flattened this way in _vectorise_controlvariate_input:
        if isinstance(val, Jaxtainer):
            output = Jaxtainer.from_array(cv_samples, val.shape, order='F')
        else:
            output = cv_samples.reshape(val.shape, order='F')
        return output

    #
    #   Reparameterisation
    #

    def _compute_joint_del_phi_reparam(self, x, theta, transform_del_phi):
        joint_del_1 = self.joint.logpdf_del_1(theta, x) # shape = (num_batch, num_samples)
        joint_del_phi = np.einsum("abj,abj...->ab...", joint_del_1, transform_del_phi)
        return joint_del_phi
    
    @staticmethod
    def _compute_approx_del_phi_reparam(approx, phi, theta, transform_del_phi):    
        approx_del_1 = approx.logpdf_del_1(theta, phi) # shape = (num_batch, num_samples, theta_dim)
        approx_del_phi = np.einsum("abj,abj...->ab...", approx_del_1, transform_del_phi)
        return approx_del_phi

class ELBO(Loss):

    _default_num_samples = {'cv': 100, 'reparam': 10}

    def __init__(self, jointdist, use_reparameterisation=False):
        self.joint = jointdist
        self.use_reparameterisation = use_reparameterisation
    
    def eval(self, approx, x, prngkey, num_samples=None):
        
        phi = approx.phi(x)

        if self.use_reparameterisation:
            loss, loss_del_phi = self._eval_elbo_reparameterisation(approx, phi, x, num_samples, prngkey)
        else:
            loss, loss_del_phi = self._eval_elbo_cv(approx, phi, x, num_samples, prngkey)

        loss_del_params = self._compute_loss_del_params(loss_del_phi, x, approx)
        loss, loss_del_params = self._avg_over_batch_dim(loss, loss_del_params)

        return loss, loss_del_params

    def _eval_elbo_reparameterisation(self, approx, phi, x, num_samples, prngkey):

        if num_samples is None:
            num_samples = self._default_num_samples['reparam']

        epsilon = approx.sample_base(num_samples, prngkey)
        theta = approx.transform(epsilon, phi)

        approx_lp = approx.logpdf(theta, phi)
        joint_lp = self.joint.logpdf(theta, x)
        loss_samples = joint_lp - approx_lp

        transform_del_phi = approx.transform_del_2(epsilon, phi)
        joint_del_phi = self._compute_joint_del_phi_reparam(x, theta, transform_del_phi)
        approx_del_phi = self._compute_approx_del_phi_reparam(approx, phi, theta, transform_del_phi)
        loss_del_phi_samples = joint_del_phi - approx_del_phi

        loss = -1*np.mean(loss_samples, axis=1)
        loss_del_phi = -1*np.mean(loss_del_phi_samples, axis=1)

        return loss, loss_del_phi
        
    def _eval_elbo_cv(self, approx, phi, x, num_samples, prngkey):

        if num_samples is None:
            num_samples = self._default_num_samples['cv']

        theta = approx.sample(num_samples, prngkey, phi) # shape = (num_batch, num_samples, theta_dim)

        approx_lp = approx.logpdf(theta, phi) # shape = (num_batch, num_samples)
        joint_lp = self.joint.logpdf(theta, x) # shape = (num_batch, num_samples)
        approx_del_phi = approx.logpdf_del_2(theta, phi) # shape = (num_batch, num_samples, *phi.shape)

        loss_samples = (joint_lp - approx_lp) # shape = (num_batch, num_samples)
        loss_del_phi_samples = np.einsum("ab,ab...->ab...", loss_samples, approx_del_phi) # shape = (num_batch, num_samples, *phi.shape)
        
        control_variate = approx_del_phi
        loss_samples = self._apply_controlvariates(loss_samples, control_variate)
        loss_del_phi_samples = self._apply_controlvariates(loss_del_phi_samples, control_variate)

        loss = -1*np.mean(loss_samples, axis=1) # shape = (num_batch,)
        loss_del_phi = -1*np.mean(loss_del_phi_samples, axis=1) # shape = (num_batch, *phi.shape)

        return loss, loss_del_phi

class SELBO(Loss):

    _default_num_samples = {'cv': 10, 'reparam': 5}

    def __init__(self, jointdist, use_reparameterisation=False):
        self.joint = jointdist
        self.use_reparameterisation = use_reparameterisation

    def eval(self, approx, x, prngkey, num_samples=None):

        phi = approx.phi(x)

        if self.use_reparameterisation:
            loss, loss_del_phi = self._eval_selbo_reparameterisation(approx, phi, x, num_samples, prngkey)
        else:
            loss, loss_del_phi = self._eval_selbo_cv(approx, phi, x, num_samples, prngkey)

        loss_del_params = self._compute_loss_del_params(loss_del_phi, x, approx)
        loss, loss_del_params = self._avg_over_batch_dim(loss, loss_del_params)

        return loss, loss_del_params

    def _eval_selbo_reparameterisation(self, approx, phi, x, num_samples, prngkey):
        
        if num_samples is None:
            num_samples = self._default_num_samples['reparam']

        epsilon = approx.sample_base(num_samples, prngkey) # shape = (num_samples, num_mixture, dim_theta)
        theta = approx.transform(epsilon, phi) # shape = (num_batch, num_samples, num_mixture, dim_theta)
    
        component_lp = approx.logpdf_components(theta, phi) # shape = (num_batch, num_samples, num_mixture)
        coeffs = approx.coefficients(phi=phi) # shape = (num_batch, num_mixture)
        loss_samples = self._compute_loss_samples(theta, x, component_lp, coeffs) # shape = (num_batch, num_samples, num_mixture)

        coeffs_del_phi = approx.coefficients_del_phi(phi=phi) # shape = (num_batch, num_mixture, phi_dims)
        transform_del_phi = approx.transform_del_2(epsilon, phi) # shape = (num_batch, num_samples, num_mixture, theta_dim, phi_dim)
        joint_del_phi = self._compute_joint_del_phi(x, theta, transform_del_phi)
        approx_del_phi = self._compute_reparam_approx_del_phi(approx, phi, theta, transform_del_phi)
        
        # Notice the mixture component dimension 'm' here:
        loss_del_phi_samples = np.einsum("am...,abm->abm...", coeffs_del_phi, loss_samples) + \
                               #np.einsum("am,abm...->abm...", coeffs, approx_del_phi)
                               np.einsum("am,ab...->ab...", coeffs, approx_del_phi)

        loss = -1*np.mean(np.einsum("am,abm->ab", coeffs, loss_samples), axis=1)
        loss_del_phi = -1*np.mean(np.sum(loss_del_phi_samples, axis=2), axis=1)

        return loss, loss_del_phi

    def _compute_joint_del_phi(self, x, theta, transform_del_phi):
        joint_del_1 = self._compute_joint(theta, x, grad=True) # shape = (num_batch, num_samples, num_mixture, theta_dim)
        joint_del_phi = np.einsum('abmi...,abmi->abm...', transform_del_phi, joint_del_1)
        return joint_del_phi

    @staticmethod
    def _compute_reparam_approx_del_phi(approx, phi, theta, transform_del_phi):
        # approx_del_1 = approx.logpdf_del_1_components(theta, phi) # shape = (num_batch, num_samples, theta_dim)
        # approx_del_phi = np.einsum("mabj,mabj...->mab...", approx_del_1, transform_del_phi)
        logpdf_del_2 = approx.logpdf_del_2(theta, phi)
        return logpdf_del_2

    #
    #   General Helper Methods
    #

    def _compute_loss_samples(self, theta, x, component_lp, coefficients):
        joint_lp = self._compute_joint(theta, x) # shape = (num_batch, num_samples, num_mixture)
        approx_lp = np.einsum("am,abm->ab", coefficients, component_lp) # shape = (num_batch, num_samples)
        # Transpose to ensure correct broadcasting:
        loss_samples = (joint_lp.T - approx_lp.T).T  # shape = (num_batch, num_samples, num_mixture)
        return loss_samples

    def _compute_joint(self, theta, x, grad=False):
        theta_shape = theta.shape
        theta, x = self._reshape_x_and_theta(theta, x)
        if grad:
            joint_val = self.joint.logpdf_del_1(theta, x) # shape = (num_batch*num_mixture, num_samples, theta_dim)
        else:
            joint_val = self.joint.logpdf(theta, x) # shape = (num_batch*num_mixture, num_samples)
        joint_val = self._reshape_joint_output(joint_val, theta_shape, grad) # shape = (num_batch, num_samples, num_mixture)
        return joint_val

    @staticmethod
    def _reshape_x_and_theta(theta, x):
        
        # Before reshaping:
            #   theta.shape = (num_batch, num_samples, num_mixture, theta_dim)
            #   x.shape = (num_batch, x_dim)
        # After reshaping:
            #   theta.shape = (num_batch*num_mixture, num_samples, theta_dim)
            #   x.shape = (num_batch*num_mixture, x_dim)
        
        num_components = theta.shape[2]
        x = jnp.tile(x, (num_components, 1)) # shape = (num_batch*num_mixture, x_dim)

        theta = jnp.swapaxes(theta, 1, 2)  # shape = (num_batch, num_mixture, num_samples, theta_dim)
        new_theta_shape = (np.prod(theta.shape[:2]), *theta.shape[2:])
        theta = theta.reshape(new_theta_shape) # shape = (num_batch*num_mixture, num_samples, theta_dim)
        
        return theta, x

    @staticmethod
    def _reshape_joint_output(joint_val, theta_shape, is_grad=False):
        # theta_shape = (num_batch, num_samples, num_mixture, theta_dim) -> (num_batch, num_mixture, num_samples, theta_dim):
        theta_shape = list(theta_shape)
        theta_shape[1], theta_shape[2] = theta_shape[2], theta_shape[1]
        if is_grad: 
            joint_val = joint_val.reshape(*theta_shape[:-1], -1) # shape = (num_batch, num_mixture, num_samples, dim_theta)
        else:
            joint_val = joint_val.reshape(theta_shape[:-1]) # shape = (num_batch, num_mixture, num_samples)
        joint_val = jnp.swapaxes(joint_val, 1, 2) # shape = (num_batch, num_samples, num_mixture)
        return joint_val

class ForwardKL(Loss):

    _default_num_samples = {'cv': 1000, 'reparam': 1000}

    def __init__(self,  jointdist=None, posterior_samples=None, use_reparameterisation=False):
        self.joint = jointdist
        self.posterior_samples = posterior_samples
        self.use_reparameterisation = use_reparameterisation

    @staticmethod
    def _compute_importance_samples(samples, approx_logpdf, joint_logpdf):
        log_wts = joint_logpdf - approx_logpdf # shape = (num_batch, num_samples)
        log_wts_max = jnp.max(log_wts, axis=1, keepdims=True) # shape = (num_batch, 1)
        unnorm_wts = jnp.exp(log_wts-log_wts_max) # shape = (num_batch, num_samples)
        unnorm_wts_sum = jnp.sum(unnorm_wts, axis=1, keepdims=True) # shape = (num_batch, 1)
        if isinstance(samples, Jaxtainer):
            numerator = np.einsum('ab,ab...->ab...', unnorm_wts, samples) # shape = (num_batch, num_samples, dim_phi)
            # Need to broadcast in 'opposite direction', requiring transposes:
            result = (numerator.T/unnorm_wts_sum.T).T # shape = (num_batch, num_samples, dim_phi)
        else:
            result = unnorm_wts*samples/unnorm_wts_sum # shape = (num_batch, num_samples)
        return result

    def eval(self, approx, x, prngkey=None, num_samples=None):

        if all(not hasattr(self, attr) for attr in ['joint', 'posterior_samples']):
            raise ValueError('Must specify either jointdist or posterior_samples.')

        phi = approx.phi(x)

        if self.posterior_samples is not None:
            loss, loss_del_phi = self._eval_posterior_samples(approx, phi)
        else:

            if prngkey is None:
                raise ValueError('Must specify prngkey to use reparameterisation or control variate methods.')

            if self.use_reparameterisation:
                loss, loss_del_phi = self._eval_reparameterisation(approx, phi, x, num_samples, prngkey)
            else:
                loss, loss_del_phi = self._eval_controlvariates(approx, phi, x, num_samples, prngkey)
            
        loss_del_params = self._compute_loss_del_params(loss_del_phi, x, approx)

        loss, loss_del_params = self._avg_over_batch_dim(loss, loss_del_params)

        return loss, loss_del_params

    def _eval_posterior_samples(self, approxdist, phi):
        approx_lp = approx.logpdf(self.posterior_samples, phi=phi)
        loss = -1*jnp.mean(approx_lp, axis=1)
        approx_del_phi = approx.logpdf_del_2(self.posterior_samples, phi=phi)
        grad = -1*np.mean(approx_del_phi, axis=1)
        return loss, grad

    def _eval_reparameterisation(self, approx, phi, x, num_samples, prngkey):
        
        if num_samples is None:
            num_samples = self._default_num_samples['reparam']

        epsilon = approx.sample_base(num_samples, prngkey)
        theta = approx.transform(epsilon, phi)

        approx_lp = approx.logpdf(theta, phi) # shape = (num_batch, num_samples)
        loss_samples = approx_lp
        
        transform_del_phi = approx.transform_del_2(epsilon, phi)
        joint_del_phi = self._compute_joint_del_phi_reparam(x, theta, transform_del_phi)
        approx_del_phi = self._compute_approx_del_phi_reparam(approx, phi, theta, transform_del_phi)

        loss_del_phi_samples = np.einsum("ab,ab...->ab...", approx_lp, joint_del_phi) + \
                               np.einsum("ab,ab...->ab...", 1-approx_lp, approx_del_phi)

        loss_samples = self._compute_importance_samples(loss_samples, approx_lp, joint_lp)
        loss_del_phi_samples = self._compute_importance_samples(loss_del_phi_samples, approx_lp, joint_lp)

        loss = -1*np.mean(loss_samples, axis=1) # shape = (num_batch,)
        loss_del_phi = -1*np.mean(loss_del_phi_samples, axis=1) # shape = (num_batch, *phi.shape)

        return loss, loss_del_phi

    def _eval_controlvariates(self, approx, phi, x, num_samples, prngkey):
        
        if num_samples is None:
            num_samples = self._default_num_samples['cv']

        theta = approx.sample(num_samples, prngkey, phi) # shape = (num_batch, num_samples, theta_dim)

        approx_lp = approx.logpdf(theta, phi)  # shape = (num_batch, num_samples)
        approx_del_phi_samples = approx.logpdf_del_2(theta, phi)

        loss_samples = approx_lp
        joint_lp = self.joint.logpdf(theta, x)
        loss_samples = self._compute_importance_samples(loss_samples, approx_lp, joint_lp)
        loss_del_phi_samples = self._compute_importance_samples(approx_del_phi_samples, approx_lp, joint_lp)

        # Apply control variates:
        control_variate = approx_del_phi_samples
        loss_samples = self._apply_controlvariates(loss_samples, control_variate)
        loss_del_phi_samples = self._apply_controlvariates(loss_del_phi_samples, control_variate)

        loss = -1*np.mean(loss_samples, axis=1) # shape = (num_batch,)
        loss_del_phi = -1*np.mean(loss_del_phi_samples, axis=1) # shape = (num_batch, *phi.shape)

        return loss, loss_del_phi

class MSE(Loss):

    def __init__(self, target=None):
        self.target = target

    # Need kwargs to 'absorb' unnecessary arguments passed by optimiser:
    def eval(self, amortised, x, **kwargs):

        phi = amortised.phi(x)
        if self.target is None:
            target_phi = amortised.distribution.phi()
        else:
            target_phi = self.target

        phi_diff = phi - target_phi
        mse = (phi_diff**2).sum_all()
        mse_del_phi = 2*phi_diff[:,None,:] # shape = (num_batch, 1, phi_shape)
        mse_del_params = self._compute_loss_del_params(mse_del_phi, x, amortised) # shape = (num_batch, param_shape)
        mse_del_params = np.mean(mse_del_params, axis=0) # shape = (param_shape,)

        return mse, mse_del_params