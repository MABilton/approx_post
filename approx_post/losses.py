import numpy as np
import jax.numpy as jnp
from arraytainers import Jaxtainer

class Loss:

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

    def _apply_controlvariates(self, val, cv):

        num_batch, num_samples = self._get_batch_and_sample_size_from_cv(cv)

        val_vec = self._vectorise_controlvariate_input(val, new_shape=(num_batch, num_samples, -1))
        cv_vec = self._vectorise_controlvariate_input(cv, new_shape=(num_batch, num_samples, -1))
        
        var = self._compute_covariance(val_1=cv_vec, val_2=cv_vec) # shape = (num_batches, num_samples, dim_cv, dim_cv)
        cov = self._compute_covariance(val_1=cv_vec, val_2=val_vec, subtract_mean_from_val_2=True) # shape = (num_batches, num_samples, dim_cv, dim_val)

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
    def _compute_covariance(val_1, val_2, subtract_mean_from_val_2=False):
        if subtract_mean_from_val_2:
            val_2 = val_2 - np.mean(val_2, axis=1, keepdims=True)
        return np.mean(np.einsum("abi,abj->abij", val_1, val_2), axis=1)
    
    @staticmethod
    def _compute_controlvariate_samples(val_vec, cv_vec, cov, var):
        a = -1*np.linalg.solve(var, cov) # shape = (num_batch, num_samples, dim_cv, dim_val)
        return val_vec + np.einsum("aij,abi->abj", a, cv_vec) # shape = (num_batch, num_samples, num_val)

    @staticmethod
    def _reshape_controlvariate_output(cv_samples, val):
        # order='F' since val flattened this way in _vectorise_controlvariate_input:
        if isinstance(val, Jaxtainer):
            output = Jaxtainer.from_array(cv_samples, val.shape, order='F')
        else:
            output = cv_samples.reshape(val.shape, order='F')
        return output

    def _compute_joint_del_phi_reparameterisation(self, x, theta, transform_del_phi):
        joint_del_1 = self.joint.logpdf_del_1(theta, x)
        return np.einsum("abj,abj...->ab...", joint_del_1, transform_del_phi)
    
    @staticmethod
    def _compute_approx_del_phi_reparameterisation(approx, phi, theta, transform_del_phi, approx_is_mixture=False):    
        if approx_is_mixture:
            approx_del_1 = approx.logpdf_del_1_components(theta, phi) # shape = (num_batch, num_samples, theta_dim)
            approx_del_phi = np.einsum("mabj,mabj...->mab...", approx_del_1, transform_del_phi)
        else:
            approx_del_1 = approx.logpdf_del_1(theta, phi) # shape = (num_batch, num_samples, theta_dim)
            approx_del_phi = np.einsum("abj,abj...->ab...", approx_del_1, transform_del_phi)
        return approx_del_phi

class ReverseKL(Loss):

    _default_num_samples = {'elbo_cv': 100, 'elbo_reparam': 10, 'selbo_cv': 10, 'selbo_reparam': 5}

    def __init__(self, jointdist, use_reparameterisation=False, method='elbo'):
        if method.lower() not in ('elbo', 'selbo'):
            raise ValueError(f"Invalid method value provided; valid values are: 'elbo' or 'selbo'")
        self.joint = jointdist
        self.use_reparameterisation = use_reparameterisation
        self.method = method
    
    def eval(self, approx, x, prngkey, num_samples=None):

        phi = approx.phi(x)

        if self.method == 'elbo':
            if self.use_reparameterisation:
                loss, loss_del_phi = self._eval_elbo_reparameterisation(approx, phi, x, num_samples, prngkey)
            else:
                loss, loss_del_phi = self._eval_elbo_cv(approx, phi, x, num_samples, prngkey)
        elif self.method == 'selbo':
            if self.use_reparameterisation:
                loss, loss_del_phi = self._eval_selbo_reparameterisation(approx, phi, x, num_samples, prngkey)
            else:
                loss, loss_del_phi = self._eval_selbo_cv(approx, phi, x, num_samples, prngkey)
        else:
            raise ValueError("Invalid method attribute value: must be either 'elbo' or 'selbo'.")

        loss_del_params = self._compute_loss_del_params(loss_del_phi, x, approx)

        loss, loss_del_params = self._avg_over_batch_dim(loss, loss_del_params)

        return loss, loss_del_params

    def _eval_elbo_reparameterisation(self, approx, phi, x, num_samples, prngkey):

        if num_samples is None:
            num_samples = self._default_num_samples['elbo_reparam']

        epsilon = approx.sample_base(num_samples, prngkey)
        theta = approx.transform(epsilon, phi)

        approx_lp = approx.logpdf(theta, phi)
        joint_lp = self.joint.logpdf(theta, x)
        loss_samples = joint_lp - approx_lp

        transform_del_phi = approx.transform_del_2(epsilon, phi)
        joint_del_phi = self._compute_joint_del_phi_reparameterisation(x, theta, transform_del_phi)
        approx_del_phi = self._compute_approx_del_phi_reparameterisation(approx, phi, theta, transform_del_phi)
        loss_del_phi_samples = joint_del_phi - approx_del_phi

        loss = -1*np.mean(loss_samples, axis=1)
        loss_del_phi = -1*np.mean(loss_del_phi_samples, axis=1)

        return loss, loss_del_phi
        
    def _eval_elbo_cv(self, approx, phi, x, num_samples, prngkey):

        if num_samples is None:
            num_samples = self._default_num_samples['elbo_cv']

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

    def _eval_selbo_reparameterisation(self, approx, phi, x, num_samples, prngkey):
        
        if num_samples is None:
            num_samples = self._default_num_samples['selbo_reparam']

        epsilon = approx.sample_base(num_samples, prngkey)
        theta = approx.transform(epsilon, phi)
    
        approx_lp = approx.logpdf(theta, phi)
        joint_lp = self.joint.logpdf(theta, x)
        loss_samples = joint_lp - approx_lp

        transform_del_phi = approx.transform_del_2(epsilon, phi)
        joint_del_phi = self._compute_joint_del_phi_reparameterisation(x, theta, transform_del_phi)
        approx_del_phi = \
            self._compute_approx_del_phi_reparameterisation(approx, phi, theta, transform_del_phi, approx_is_mixture=True)
        # Notice the mixture component dimension 'm' here:
        loss_del_phi_samples = np.einsum("abi,mabi...->mab...", joint_del_theta, transform_del_phi) \
                             - np.einsum("mabi,mabi...->mab...", approx_del_phi, transform_del_phi)

        coefficients = approx.coefficients(phi=phi)
        loss = np.mean(np.einsum("m, mab->ab", coefficients, loss_samples), axis=1)
        loss_del_phi = np.einsum("m, mab->ab", coefficients, loss_del_phi_samples)

        return loss, loss_del_phi

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
        joint_del_phi = self._compute_joint_del_phi_reparameterisation(x, theta, transform_del_phi)
        approx_del_phi = self._compute_approx_del_phi_reparameterisation(approx, phi, theta, transform_del_phi)

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