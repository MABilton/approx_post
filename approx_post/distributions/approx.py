import json
from json import JSONDecodeError
import os
import numpy as np

from .gaussian import create_gaussian

class ApproximateDistribution:

    @classmethod
    def gaussian(cls, ndim, params=None, mean_lb=None, mean_ub=None, var_ub=None, cov_lb=None, cov_ub=None):
        
        # Create functions and attributes to pass to constructor:
        func_dict, attr_dict = \
            create_gaussian(ndim, mean_lb, mean_ub, var_ub, cov_lb, cov_ub)

        if params is not None:
            attr_dict['params'] = params

        # Attribute values to store with saved distribution:
        save_dict = {'constructor': 'gaussian',
                     'args': (ndim,),
                     'kwargs': {'mean_lb': mean_lb, 'mean_ub': mean_ub,
                                'var_ub': var_ub, 'cov_lb': cov_lb, 'cov_ub': cov_ub}}

        return cls(func_dict, attr_dict, save_dict)

    def __init__(self, func_dict, attr_dict, save_dict):
        self._func_dict = func_dict

        def phi_func(params,x):
            return np.repeat(params[None,:], x.shape[0], axis=0)

        self._func_dict['phi'] = phi_func
        self._save_dict = save_dict
        self._attr_dict = attr_dict

    @property
    def params(self):
        return self._attr_dict['params']
    @property
    def phi_shape(self):
        return self._attr_dict['phi_shape']
    @property
    def phi_lb(self):
        return self._attr_dict['phi_lb']
    @property
    def phi_ub(self):
        return self._attr_dict['phi_ub']

    def initialise_params():
        pass

    def phi(self, x):
        return self._func_dict['phi'](x, self.params)

    def logpdf(self, theta, x=None):
        if self.phi is not None:
            lp = self._fun_dict['lp'](theta, self.phi(x))
        else:
            error_message = '''Need to assign parameters phi to distribution 
                               before computing log-probabilities.'''
            raise ValueError(error_message)
        return lp

    def sample(self, num_samples, x=None): 
        if self.phi is not None:
            try:
                samples = self._fun_dict['sample'](num_samples, self.phi(x))
            except KeyError:
                epsilon_samples = self._fun_dict['sample_base'](num_samples)
                samples = self._fun_dict['transform'](epsilon_samples, self.phi(x))
        else:
            error_message = '''Need to assign parameters phi to 
                               distribution before sampling.'''
            raise ValueError(error_message)
        return samples  
    
    def __repr__(self):
        return f'{self.__class__.__name__}'

    @classmethod
    def load(cls, load_dir):
        load_dict = cls._load_json(load_dir)
        constructor = getattr(cls, load_dict['constructor'])
        return constructor(*load_dict['args'], **load_dict['kwargs'])

    @classmethod
    def _load_json(cls, load_dir):
        # First try load the JSON file:
        try:
            with open(load_dir, 'r') as f:
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

    def save(self, save_name, save_dir='.', indent=4):
        to_save = self._save_dict.copy()
        to_save['params'] = list(self.params)
        save_json = json.dumps(to_save, indent)
        save_dir = os.path.join(save_dir, save_name)
        with open(save_dir, 'w') as f:
            f.write(save_json)