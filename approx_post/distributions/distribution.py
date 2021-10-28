import json
from json import JSONDecodeError


class Distribution:
    def __repr__(self):
        try:
            repr_str = f'{self.__class__.__name__} object with parameters {repr(self.phi)}'
        except AttributeError:
            repr_str = f'{self.__class__.__name__} object with no specified parameters'
        return repr_str

    @classmethod
    def _load_json(cls, load_dir):
        try:
            with f as open(load_dir, 'r'):
                dist_json = json.read(f)
        except FileNotFoundError:
            file_dir, file_name
            error_msg = f'File {file_dir} not found in directory {file_dir}.'
            raise FileNotFoundError(error_msg)
        except JSONDecodeError:
            error_msg = f'''Unable to decode contents of {file_name}. Ensure that
                            {file_name} is in a JSON-readable format.'''
            raise JSONDecodeError(error_msg)
        
        return dist_json

    @classmethod
    def _load_distribution(cls, load_dir, fun_to_load, attr_to_load):
        dist_json = self._load_json(load_dir)


        for attr in attr_to_load:

    def _save_distribution(self, save_name, fun_to_save, attr_to_save, save_dir='.', indent=4):
        
        save_dict = {}
        
        for key, fun in fun_to_save.items():
            try:
                fun_lines = inspect.getsource(fun)
                save_dict[key] = fun_lines
            except TypeError:
                error_msg = f'''Unable to read source code of {key} function {fun.__name__}.
                                Make sure that {fun.__name__} is a Python-implemented function.'''
                raise TypeError(error_msg)

        for key, attr in attr_to_save.items():
            # Check that attribute is JSON-serialisable:
            try:
                json.dumps({key: attr})
                save_dict[key] = attr
            except (TypeError, OverflowError):
                error_msg = f'''The {key} attribute with value {attr} is not
                                JSON-serialisable and, thus, cannot be saved.'''
                raise TypeError(error_msg)
        
        save_json = json.dumps(save_dict, indent)



class JointDistribution(Distribution):

    @classmethod
    def from_joint(lp, sample, x, lp_grad=None):
        func_dict = {'sample': sample, 'lp': lp}
        if lp_grad is not None:
            func_dict['lp_del_1'] = lp_grad
        return cls(func_dict, x)

    @classmethod
    def from_prior_and_likelihood(cls, prior_lp, prior_sample, likelihood_lp, likelihood_sample, 
                                  prior_grad=None, likelihood_grad=None):
        
        # Create joint functions from prior and likelihood:
        lp, sample = create_joint(prior_lp, prior_sample, likelihood_lp, likelihood_sample)
        lp_del_theta = create_joint_grad(prior_grad, likelihood_grad)

        # Place functions in dictionary and create clas:
        func_dict = {'sample': sample, 'lp': lp}
        if lp_del_theta is not None:
            func_dict['lp_del_1'] = lp_del_theta
        return cls(func_dict, x)

    def __init__(self, func_dict, x):
        self.x = x
        self._funct_dict = {'lp': lp}
        if lp_del_1 is not None:
            self._funct_dict['lp_del_1'] = lp_del_1
            
    def log_prob(self, theta, x=None):
        x_obs = self.x if x is None
        lp = self.fun_dict['lp'](theta, x_obs)
        return lp

    def sample(self, num_samples):
        
class ApproximateDistribution(Distribution):

    @classmethod
    def using_posterior_samples(cls, lp, phi_shape, phi_lb=None, phi_ub=None):
        # Place 'base' functions in dictionary (for purposes of saving the dist):
        base_funcs = {'lp': lp}
        
        lp_vmap = vectorise_functions(lp)
        lp_del_phi = create_approximation_grads(lp) 
        func_dict = {'lp': lp, 'lp_del_phi': lp_del_phi}
        return cls(func_dict, phi_shape, phi_lb, phi_ub, base_funcs)

    @classmethod
    def using_control_variates(cls, lp, sample, phi_shape, phi_lb=None, phi_ub=None):
        # Place 'base' functions in dictionary (for purposes of saving the dist):
        base_funcs = {'lp': lp, 'sample': sample}

        lp_vmap = vectorise_functions(lp)
        lp_del_phi = create_approximation_grads(lp) 
        func_dict = {'sample': sample,
                     'lp': lp, 'lp_del_phi': lp_del_phi}
        return cls(func_dict, phi_shape, phi_lb, phi_ub, base_funcs)
    
    @classmethod 
    def using_reparameterisation(cls, lp, sample_base, transform, 
                                 phi_shape, phi_lb=None, phi_ub=None):
        
        # Place 'base' functions in dictionary (for purposes of saving the dist):
        base_funcs = {'lp': lp, 'sample': sample}
        
        # Create new functions:
        lp_vmap, transform_vmap = vectorise_functions(lp, transform)
        lp_del_1, lp_del_2, transform_del_phi = create_approximation_grads(lp, transform)

        # Place in dictionary:
        func_dict = {'sample_base': sample_base,
                     'transform': transform, 'transform_del_phi': transform_del_phi
                     'lp': lp, 'lp_del_1': lp_del_1, 'lp_del_2': lp_del_2}
        
        return cls(func_dict, phi_shape, phi_lb, phi_ub, base_funcs)

    def __init__(self, func_dict, phi_shape, phi_lb=None, phi_ub=None, func_to_save=None):
        self.phi_shapes = phi_shapes
        self._func_dict = func_dict
        self.phi_lb = phi_lb
        self.phi_ub = phi_ub
        self._func_to_save = func_to_save

    def set_phi(self, phi):
        self.phi = phi

    def log_prob(self, theta):
        try:
            lp = self._fun_dict['lp'](theta, self.phi)
        except AttributeError:
            error_message = '''Need to assign parameters phi to distribution 
                               before computing log-probabilities.'''
            raise ValueError(error_message)
        return lp

    def sample(self, num_samples): 
        try:
            samples = self._fun_dict['sample'](num_samples, self.phi)
        except KeyError:
            epsilon_samples = self._fun_dict['sample_base'](num_samples)
            samples = self._fun_dict['transform'](epsilon_samples, self.phi)
        except AttributeError:
            error_message = '''Need to assign parameters phi to 
                                distribution before sampling.'''
            raise ValueError(error_message)
        return samples

    def save(self):
        # First, get 

    