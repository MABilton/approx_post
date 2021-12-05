import numpy as np
from jaxlib.xla_extension import DeviceArray
from itertools import chain
from more_itertools.more import always_iterable

class ArrayContainer(np.lib.mixins.NDArrayOperatorsMixin):

    ARRAY_TYPES = (np.ndarray, DeviceArray)

    def __init__(self, contents, containerise_values=True):
        self.contents = contents
        self._type = dict if hasattr(self.contents, 'keys') else list     
        # If nested dictionaries, convert all of these:
        if containerise_values:
          to_covert = [key for key in self.keys() 
                       if hasattr(self[key], 'keys') and not issubclass(type(self[key]), ArrayContainer)]
          for key in to_covert:
              self[key] = self.__class__(self[key])
                    
    def __len__(self):
        return len(self.contents)

    def copy(self):
      return self.__class__(self.contents.copy())

    # String representating of container object:
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.contents)})"

    def keys(self):
      if self._type is dict:
        keys = [key for key in self.contents.keys()]
        keys = self.sort_keys(keys)
      else:
        keys = tuple(i for i in range(len(self.contents)))
      return keys
    
    def values(self):
      return tuple(self[key] for key in self.keys())

    def items(self):
      return tuple((key, self[key]) for key in self.keys())

    # Need to sort keys for sake of key comparisons:
    def sort_keys(self, keys):
      types_in_list = [(type(key).__name__) for key in keys]
      types_in_list = list(set(types_in_list))
      types_in_list = sorted(types_in_list)
      sorted_sublists = [sorted([key for key in keys if type(key).__name__ == type_i]) for type_i in types_in_list]
      sorted_keys = tuple(chain.from_iterable(sorted_sublists))
      return sorted_keys

    @property
    def shape(self):
      shapes = [self[key].shape for key in self.keys()]
      if self._type is dict:
        shapes = dict(zip(self.keys(), shapes))
      return shapes

    @property
    def shape_container(self):
      shapes = self.shape
      return self.__class__(shapes)

    @property
    def unpacked(self):
      output = [val.unpacked if issubclass(type(val), ArrayContainer)
                else val
                for val in self.values()]
      if self._type is dict:
        output = dict(zip(self.keys(), output))
      return output

    # Indexing functions:
    def __getitem__(self, key):
      key_type = type(key) 

      # If indexing with an array container:
      if issubclass(key_type, ArrayContainer):
        item = self._index_with_container(key)

      # If indexing with an array:
      elif key_type in self.ARRAY_TYPES:
        item = self._index_with_array(key)

      # If indexing using a slice or tuple of slices:
      elif all(isinstance(val, slice) for val in always_iterable(key)):
        item = self._index_with_slices(key)

      # If we're given a tuple of integers:
      elif isinstance(key, tuple) and all(isinstance(val, int) for val in always_iterable(key)):
        self._set_with_slices(key, new_value)

      # Index using a regular hash:
      else:
        item = self._index_with_hash(key)

      return item

    def _index_with_container(self, key_container):
      item = self.copy()
      for container_key in self.keys():
        array_key = key_container[container_key]
        item[container_key] = self.contents[container_key][self.array(array_key)]
      return item
  
    def _index_with_array(self, array_key):
      item = self.copy()
      for container_key in self.keys():
        item[container_key] = self.contents[container_key][self.array(array_key)]
      return item

    def _index_with_slices(self, slices):
      item = self.copy()
      for container_key in self.keys():
        item[container_key] = self.contents[container_key][slices]
      return item

    def array(self, in_array):
      return TypeError

    def _index_with_hash(self, key):
        try:
          item = self.contents[key]
        # Hash passed could be an integer instead of a key (e.g. by a Numpy function)
        except KeyError:
          keys = self.keys()
          item = self.contents[keys[key]]
        return item

    # Setting functions:
    def __setitem__(self, key, new_value):
      key_type = type(key)
      if issubclass(key_type, ArrayContainer):
        self._set_with_container(key, new_value)
      elif key_type in self.ARRAY_TYPES:
        self._set_with_array(key, new_value)
      elif all(isinstance(val, slice) for val in always_iterable(key)):
        self._set_with_slices(key, new_value)
      # If we're given a tuple of integers:
      elif isinstance(key, tuple) and all(isinstance(val, int) for val in always_iterable(key)):
        self._set_with_slices(key, new_value)
      else:
        self._set_with_hash(key, new_value)    

    def _set_with_container(self, container, new_value):
      value_is_container = issubclass(type(new_value), ArrayContainer)
      for container_key in self.keys():
        idx = container[container_key]
        value_i = new_value[container_key] if value_is_container else new_value
        self._set_array_item(container_key, idx, value_i)

    def _set_with_array(self, array_key, new_value):
      value_is_container = issubclass(type(new_value), ArrayContainer)
      for container_key in self.keys():
        value_i = new_value[container_key] if value_is_container else new_value
        self._set_array_item(container_key, array_key, value_i)

    def _set_with_slices(self, slices, new_value):
      value_is_container = issubclass(type(new_value), ArrayContainer)
      for container_key in self.keys():
        value_i = new_value[container_key] if value_is_container else new_value
        self._set_array_item(container_key, slices, value_i)

    def _set_array_item(self, key, idx, value_i):
      raise KeyError

    def _set_with_hash(self, key, new_value):
      try:
        self.contents[key] = new_value
      except IndexError:
          if self._type is list:
              error_msg = "Unable to assign items to a list-like container; use the append method instead."
              raise TypeError(error_msg)
          else:
              error_msg = f"Provided key {key} is invalid."
              raise IndexError(error_msg)

    def append(self, new_val):
        try:
            self.contents.append(new_val)
        except AttributeError:
            error_msg = """Unable to append values to dictionary-like container; 
                          use 'my_container[new_key] = new_value' instead."""
            raise AttributeError(error_msg)

    def sum(self):
      for i, key in enumerate(self.keys()):
        if i == 0:
          sum_results = self[key].copy()
        else:
          sum_results += self[key]
      return sum_results

    # numpy.all() and numpy.any() equivalents:
    def all(self):
      for key in self.keys():
        if self.contents[key].all():
          continue
        else:
          return False
      return True

    def any(self):
      for key in self.keys():
        if self.contents[key].any():
          return True
      return False

    # Numpy functions:
    def __array_function__(self, func, types, args, kwargs):
        fun_return = self._manage_function_call(func, types, *args, **kwargs)
        return fun_return

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        fun_return = self._manage_function_call(ufunc, method, *args, **kwargs)
        return fun_return

    def _manage_function_call(self, func, types, args, kwargs):
      raise AttributeError("The manage_function_call method is not implemented in " + \
                           "the base ArrayContainer class. Instead, use either the " + \
                           "NumpyContainer or JaxContainer sub-classes")

    # Helper functions used by manage_function_call in NumpyContainer and JaxContainer:
    def _prepare_args(self, args, key):
      args = [arg_i[key] if issubclass(type(arg_i), ArrayContainer) else arg_i for arg_i in args]
      return args
  
    def _prepare_kwargs(self, kwargs, key):
      kwargs = {key_i: (val_i[key] if issubclass(type(val_i), ArrayContainer) else val_i)
                for key_i, val_i in kwargs.items()}
      return kwargs
  
    def _check_container_compatability(self, args, kwargs):
      
      arg_containers =  [arg_i for arg_i in args if issubclass(type(arg_i), ArrayContainer)]
      kwarg_containers = [val_i for val_i in kwargs.values() if issubclass(type(val_i), ArrayContainer)]
      container_list = arg_containers + kwarg_containers

      if container_list:
        # Get keys, type, and length of first container:
        container_0 = container_list[0]
        keys_0 = container_0.keys()
        type_0 = container_0._type
        len_0 = len(container_0)

        for container_i in container_list[1:]:
          # Ensure containers are either all dict-like or all list-like:
          if container_i._type != type_0:
            error_msg = '''Containers being combined through operations must 
                            be all dictionary-like or all list-like.'''
            raise ValueError(error_msg)

          # Ensure containers are all of same length:
          if len(container_i) != len_0:
              error_msg = '''Containers being combined through operations must 
                             all contain the same number of elements.'''
              raise ValueError(error_msg)
          
          # Ensure containers have same keys:
          if container_i.keys() != keys_0:
            error_msg = '''Containers being combined through operations must
                           have identical sets of keys.'''
            raise KeyError(error_msg)