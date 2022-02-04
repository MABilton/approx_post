import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

class JointDistribution:

    def __init__(self, logpdf, logpdf_del_1=None):
        self._func_dict = {'logpdf': logpdf}
        if logpdf_del_1 is not None:
            self._func_dict['logpdf_del_1'] = logpdf_del_1

    def logpdf(self, theta, x):
        num_batch, num_samples = theta.shape[0:2]
        return self._func_dict['logpdf'](theta, x).reshape(num_batch, num_samples)
    
    def logpdf_del_1(self, theta, x):
        num_batch, num_samples = theta.shape[0:2]
        return self._func_dict['logpdf_del_1'](theta, x).reshape(num_batch, num_samples, -1)

class ModelPlusGaussian(JointDistribution):
    
    def __init__(self, model, noise_cov, prior_mean, prior_cov, model_grad=None):
        self.noise_cov, self.prior_mean, self.prior_cov = noise_cov, prior_mean, prior_cov
        self.model, self.model_grad = model, model_grad
        self._model_funcs = self._create_model_funcs()  
        self._logpdf, self._logpdf_del_1 = self._create_logpdf(), self._create_logpdf_del_1()
        super().__init__(self._logpdf, self._logpdf_del_1)
    
    @property
    def x_dim(self):
        return self.noise_cov.shape[0]
    
    @property
    def theta_dim(self):
        return len(self.prior_mean)

    def _create_model_funcs(self):

        def wrapped_model(theta):
            output = jnp.array(self.model(theta))
            num_batch, num_samples = theta.shape[0:2]
            return output.reshape(num_batch, num_samples, self.x_dim)
        
        def wrapped_model_grad(theta):
            output = jnp.array(self.model_grad(theta))
            num_batch, num_samples = theta.shape[0:2]
            return output.reshape(num_batch, num_samples, self.x_dim, self.theta_dim)

        return {'model': wrapped_model, 'model_grad': wrapped_model_grad}

    def _create_logpdf(self):
        
        mvn_vmap_mean = jax.vmap(mvn.logpdf, in_axes=(None,1,None), out_axes=1)

        def logpdf(theta, x):
            prior_logpdf = mvn.logpdf(theta, self.prior_mean, self.prior_cov) # shape = (num_batch, num_samples)
            x_pred = self._model_funcs['model'](theta) # shape = (num_batch, num_samples, x_dim)
            like_logpdf = mvn_vmap_mean(x, x_pred, self.noise_cov) # shape = (num_batch, num_samples, num_obs)
            return prior_logpdf + jnp.sum(like_logpdf, axis=-1) # shape = (num_batch, num_samples)
        
        return logpdf
    
    def _create_logpdf_del_1(self):
        
        mvn_del_mean = jax.jacfwd(mvn.logpdf, argnums=1)
        # Vectorise over batch dimension of theta and mean, and over the sample dimension of the mean:
        mvn_del_mean_vmap = jax.vmap(jax.vmap(mvn_del_mean, in_axes=(None,0,None)), in_axes=(0,0,None))

        def logpdf_del_1(theta, x):
            prior_del_1 = mvn_del_mean(theta, self.prior_mean, self.prior_cov) # shape = (num_batch, num_samples, theta_dim)
            x_pred = self._model_funcs['model'](theta) # shape = (num_batch, num_samples, x_dim)
            like_del_mean = mvn_del_mean_vmap(x, x_pred, self.noise_cov) # shape = (num_batch, num_samples, num_obs)
            mean_del_1 = self._model_funcs['model_grad'](theta) # shape = (num_batch, num_samples, x_dim, theta_dim)
            like_del_1 = jnp.einsum('abji,abj->abi', mean_del_1, like_del_mean) # shape = (num_batch, num_samples, theta_dim)
            return prior_grad + like_grad

        return logpdf_del_1

class PriorAndLikelihood(JointDistribution):

    def __init__(self, prior, likelihood, prior_del_1=None, like_del_1=None):
        self.prior, self.likelihood = prior, likelihood
        self.prior_del_1, self.likelihood_del_1 = prior_del_1, like_del_1
        self._logpdf, self._logpdf_del_1 = self._create_logpdf(), self._create_logpdf_del_1()
        super().__init__(self._logpdf, self._logpdf_del_1)

    def _create_logpdf(self):
        
        def logpdf(theta, x):
            num_batch, num_samples = theta.shape[0:2]
            return self.prior(theta).reshape(num_batch, num_samples) \
                    + self.likelihood(theta, x).reshape(num_batch, num_samples)
        
        return logpdf

    def _create_logpdf_del_1(self):

        def logpdf_del_1(theta, x):
            num_batch, num_samples = theta.shape[0:2]
            return self.prior_del_1(theta).reshape(num_batch, num_samples, -1) \
                    + self.likelihood_del_1(theta, x).reshape(num_batch, num_samples, -1)

        return logpdf_del_1