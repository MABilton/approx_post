import json
from json import JSONDecodeError
import os

from .gaussian import create_gaussian
from ..containers.jax import JaxContainer

class ApproximateDistribution:

    @classmethod
    def gaussian(cls, ndim, phi=None, mean_lb=None, mean_ub=None, var_ub=None, cov_lb=None, cov_ub=None):
        
        # Create functions and attributes to pass to constructor:
        func_dict, attr_dict = \
            create_gaussian(ndim, mean_lb, mean_ub, var_ub, cov_lb, cov_ub)

        if phi is not None:
            attr_dict['phi'] = phi

        # Attribute values to store with saved distribution:
        save_dict = {'constructor': 'gaussian',
                     'args': (ndim,),
                     'kwargs': {'mean_lb': mean_lb, 'mean_ub': mean_ub,
                                'var_ub': var_ub, 'cov_lb': cov_lb, 'cov_ub': cov_ub}}

        return cls(func_dict, attr_dict, save_dict)

    def __init__(self, func_dict, attr_dict, save_dict):
        # func_dict = 
        self._func_dict = func_dict
        self._save_dict = save_dict
        self.phi = attr_dict['phi']
        self.phi_shape = attr_dict['phi_shape']
        self.phi_lb = attr_dict['phi_lb']
        self.phi_ub = attr_dict['phi_ub']

    def logpdf(self, theta):
        if self.phi is not None:
            lp = self._fun_dict['lp'](theta, self.phi)
        else:
            error_message = '''Need to assign parameters phi to distribution 
                               before computing log-probabilities.'''
            raise ValueError(error_message)
        return lp

    def sample(self, num_samples): 
        if self.phi is not None:
            try:
                samples = self._fun_dict['sample'](num_samples, self.phi)
            except KeyError:
                epsilon_samples = self._fun_dict['sample_base'](num_samples)
                samples = self._fun_dict['transform'](epsilon_samples, self.phi)
        else:
            error_message = '''Need to assign parameters phi to 
                               distribution before sampling.'''
            raise ValueError(error_message)
        return samples  
    
    def __repr__(self):
        try:
            repr_str = f'{self.__class__.__name__} object with parameters {repr(self.phi)}'
        except AttributeError:
            repr_str = f'{self.__class__.__name__} object with no specified parameters'
        return repr_str

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
        to_save['phi'] = list(self.phi)
        save_json = json.dumps(to_save, indent)
        save_dir = os.path.join(save_dir, save_name)
        with open(save_dir, 'w') as f:
            f.write(save_json)