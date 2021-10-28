import numpy as np

class ArrayContainer(np.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, contents):

        self.contents = contents

        # If contents are dictionary-like:
        try:
          self.key_vals = tuple(self.contents.keys())
          self._type = dict
        # If contents are list-like:
        except AttributeError:
          self.key_vals = tuple(x for x in range(len(self.contents)))
          self._type = list
                    
    # Called when len(my_array_container) is executed:
    def __len__(self):
        return len(self.contents)

    # Returns shape of arrays:
    def shape(self):
      return np.shape(self)

    def keys(self):
      return self.key_vals

    # Methods which return iterables:
    def __iter__(self):
      return (self.contents[key_i] for key_i in self.keys())

    # Called when my_array_container[key] is executed:
    def __getitem__(self, key):
        # If correct key is passed:
        try:
          item = self.contents[key]
        # If an integer is passed to a dictionary data-type 
        # (potentially by a Numpy function):
        except KeyError:
          item = self.contents[self.key_vals[key]]
        return item

    # Called when 'my_array_container[new_key] = new_value' is executed
    def __setitem__(self, key, new_value):
        try:
          self.contents[key] = new_value
          # Update key-list if this is a new key:
          if key not in self.keys():
            self._add_new_key(key)
        except IndexError:
            if self._type is list:
                raise TypeError("Unable to assign items to a list-like container; use the append method instead.")
            else:
                raise IndexError(f"Provided key {key} is invalid.")
          

    # Appends item to end of list-like container:
    def append(self, new_val):
        try:
            self.contents.append(new_val)
            new_key = len(self.keys())
            self._add_new_key(new_key)
        except AttributeError:
            raise AttributeError("Unable to append values to dictionary-like " + \
                                 "container; use 'my_container[new_key] = " + \
                                 "new_value' instead.")

    # Helper function to add new key to keys tuple attribute:
    def _add_new_key(self, new_key):
        old_keys = list(self.keys())
        self.key_vals = tuple(old_keys + [new_key])

    # String representating of container object:
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.contents)})"

    # Called when container passed to NumPy function:
    def __array_function__(self, func, types, args, kwargs):
        fun_return = self.manage_function_call(func, types, *args, **kwargs)
        return fun_return

    # Called when container passed to NumPy universal function:
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        fun_return = self.manage_function_call(ufunc, method, *args, **kwargs)
        return fun_return

    def manage_function_call(func, types, args, kwargs):
      raise AttributeError("The manage_function_call method is not implemented in " + \
                           "the base ArrayContainer class. Instead, use either the " + \
                           "NumpyContainer or JaxContainer sub-classes")

    # Helper functions used by manage_function_call in NumpyContainer and JaxContainer:
    def find_containers_in_args(self, args):
      container_list, arg_list = [], []
      for arg_i in args:
        container_list.append(arg_i) if type(arg_i)==self.__class__ else arg_list.append(arg_i)
      return (container_list, arg_list)

    def check_container_compatability(self, container_list):
      
      # Get keys, type, and length of first container:
      container_0 = container_list[0]
      keys_0 = container_0.keys()
      type_0 = container_0._type
      len_0 = len(container_0)

      for container_i in container_list[1:]:
        # Ensure containers are either all dict-like or all list-like:
        if container_i._type != type_0:
          raise ValueError('Containers being combined through operations must ' + \
                           'be all dictionary-like or all list-like.')

        # Ensure containers are all of same length:
        if len(container_i) != len_0:
            raise ValueError('Containers being combined through operations must ' + \
                             'all contain the same number of elements.')
        
        # Ensure containers have same keys:
        if container_i.key_vals != keys_0:
          raise KeyError('Containers being combined through operations must ' + \
                         'have identical sets of keys.')

class NumpyContainer(ArrayContainer):

    def __init__(self, contents, convert_inputs=True):
        super().__init__(contents)
        if convert_inputs:
          self.convert_contents_to_numpy()

    def convert_contents_to_numpy(self):
        for i, key_i in enumerate(self.keys()):
            contents_i = self.contents[key_i]
            try:
                self.contents[key_i] = np.array(contents_i)
            except TypeError:
                raise TypeError(f"Element {i} of type {type(contents_i)} " + \
                                 "cannot be converted to numpy array.")
                
    def manage_function_call(self, func, types, *args, **kwargs):

      output_dict = {}

      # Identify all instances of containers in args:
      container_list, arg_list = self.find_containers_in_args(args)
      if container_list:
        self.check_container_compatability(container_list)

      for key in self.keys():
        kwargs_dict = self.find_containers_in_kwargs(kwargs, key)
        contents_i = [cont[key] for cont in container_list]
        output_dict[key] = func(*contents_i, *arg_list, **kwargs_dict)
      
      if self._type is list:
        output_list = list(output_dict.values())
        output_container = NumpyContainer(output_list)
      else:
          output_container = NumpyContainer(output_dict)

      return output_container

    # Helper functions used by __array_ufunc__ and __array_function__:
    def find_containers_in_kwargs(self, kwargs, key):
      try:
        kwargs_dict = kwargs.copy()
        kwargs_dict['out'] = tuple([container[key] for container in kwargs['out']])
      except KeyError:
        kwargs_dict = kwargs
      return kwargs_dict