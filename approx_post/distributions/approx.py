from .distribution import Distribution

class ApproximateDistribution(Distribution):

    _func_to_load = ('sample', 'sample_base', 'lp', 'transform', 
                     'transform_del_phi', 'lp_del_1', 'lp_del_2')
    _attr_to_load = ('phi_shape', 'phi_lb', 'phi_ub')

    @classmethod
    def using_posterior_samples(cls, lp, phi_shape, phi_lb=None, phi_ub=None):
        # Place 'base' functions in dictionary (for purposes of saving the dist):
        func_to_save = {'lp': lp}
        lp_vmap = vectorise_functions(lp)
        lp_del_phi = create_approximation_grads(lp) 
        func_dict = {'lp': lp, 'lp_del_phi': lp_del_phi}
        attr_dict = {'phi_shape': phi_shape, 'phi_lb': phi_lb, 'phi_ub': phi_ub}
        return cls(func_dict, attr_dict, func_to_save)

    @classmethod
    def using_control_variates(cls, lp, sample, phi_shape, phi_lb=None, phi_ub=None):
        # Place 'base' functions in dictionary (for purposes of saving the dist):
        func_to_save = {'lp': lp, 'sample': sample}

        lp_vmap = vectorise_functions(lp)
        lp_del_phi = create_approximation_grads(lp) 
        func_dict = {'sample': sample,
                     'lp': lp, 'lp_del_phi': lp_del_phi}
        attr_dict = {'phi_shape': phi_shape, 'phi_lb': phi_lb, 'phi_ub': phi_ub}
        return cls(func_dict, attr_dict, func_to_save)
    
    @classmethod 
    def using_reparameterisation(cls, lp, sample_base, transform, 
                                 phi_shape, phi_lb=None, phi_ub=None):
        
        # Place 'base' functions in dictionary (for purposes of saving the dist):
        func_to_save = {'lp': lp, 'sample': sample}
        
        # Create new functions:
        lp_vmap, transform_vmap = vectorise_functions(lp, transform)
        lp_del_1, lp_del_2, transform_del_phi = create_approximation_grads(lp, transform)

        # Place in dictionary:
        func_dict = {'sample_base': sample_base,
                     'transform': transform, 'transform_del_phi': transform_del_phi
                     'lp': lp, 'lp_del_1': lp_del_1, 'lp_del_2': lp_del_2}     
        attr_dict = {'phi_shape': phi_shape, 'phi_lb': phi_lb, 'phi_ub': phi_ub}

        return cls(func_dict, attr_dict, func_to_save)

    def __init__(self, func_dict, attr_dict, func_to_save=None):
        super().__init__(func_dict, attr_dict)
        self.phi = None
        self.phi_shape = attr_dict['phi_shape']
        self.phi_lb = attr_dict['phi_lb']
        self.phi_ub = attr_dict['phi_ub']
        self._func_to_save = func_to_save
        
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
    
    @classmethod
    def load(cls, load_dir):
        dist_json = cls._load_json(load_dir)
        fun_dict, attr_dict = cls._load_distribution(dist_json)
        return cls(func_dict, attr_dict)

    @classmethod
    def _load_json(cls, load_dir):
        # First try load the JSON file:
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
    def _load_distribution(cls, dist_json):
        func_dict = {}
        for i, fun_name in enumerate(cls._func_to_load):
            try:
                func_dict[fun_name] = load_dict[fun_name]
            except KeyError:
                continue
        attr_dict = {}
        for i, attr_name in enumerate(cls._attr_to_load):
            try:
                attr_dict[attr_name] = load_dict[attr_name]
            except KeyError:
                continue
        return (fun_dict, attr_dict)

    def save(self, save_name, save_dir='.'):
        self._save_distribution(save_name, save_dir, self._fun_to_save, self._attr_to_save)

    def _save_distribution(self, save_name, save_dir, fun_to_save, attr_to_save, indent=4):
        
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

        save_dir = os.path.join(save_dir, save_name)
        with open(save_dir, 'w') as f:
            f.write(save_json)