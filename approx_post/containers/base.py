import numpy as np
import jaxlib

class ArrayContainer(np.lib.mixins.NDArrayOperatorsMixin):

    ARRAY_TYPES = (np.ndarray, jaxlib.xla_extension.DeviceArray)

    def __init__(self, contents):
        self.contents = contents
        # If contents are dictionary-like:
        try:
          self._keys = tuple(self.contents.keys())
          self._type = dict
        # If contents are list-like:
        except AttributeError:
          self._keys = tuple(x for x in range(len(self.contents)))
          self._type = list
                    
    def __len__(self):
        return len(self.contents)

    def copy(self):
      return self.__class__(self.contents.copy())

    # String representating of container object:
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.contents)})"

    def keys(self):
      return self._keys

    def shape(self):
      shapes = [self[key].shape for key in self.keys()]
      if self._type is dict:
        shapes = dict(zip(self.keys(), shapes))
      return shapes

    # Indexing functions:
    def __getitem__(self, key):
      key_type = type(key)
      if issubclass(key_type, ArrayContainer):
        item = self._index_with_container(key)
      elif key_type in self.ARRAY_TYPES:
        item = self._index_with_array(key)
      else:
        item = self._index_with_value(key)
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

    def array(self, in_array):
      return TypeError

    def _index_with_value(self, key):
        try:
          item = self.contents[key]
        except KeyError:
          item = self.contents[self._keys[key]]
        return item

    # Setting functions:
    def __setitem__(self, key, new_value):
      key_type = type(key)
      if issubclass(key_type, ArrayContainer):
        self._set_with_container(key, new_value)
      elif key_type in self.ARRAY_TYPES:
        self._set_with_array(key, new_value)
      else:
        self._set_with_value(key, new_value)    

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

    def _set_array_item(self, key, idx, value_i):
      raise KeyError

    def _set_with_value(self, key, new_value):
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

    def append(self, new_val):
        try:
            self.contents.append(new_val)
            new_key = len(self.keys())
            self._add_new_key(new_key)
        except AttributeError:
            raise AttributeError("Unable to append values to dictionary-like " + \
                                 "container; use 'my_container[new_key] = " + \
                                 "new_value' instead.")

    def _add_new_key(self, new_key):
        old_keys = list(self._keys)
        self._keys = tuple(old_keys + [new_key])

    # numpy.all() and numpy.any() equivalents:
    def all(self):
      for key in self.keys():
        if np.all(self.contents[key]):
          continue
        else:
          return False
      return True

    def any(self):
      for key in self.keys():
        if np.any(self.contents[key]):
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
        keys_0 = sorted(container_0.keys())
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
          if sorted(container_i.keys()) != keys_0:
            raise KeyError('Containers being combined through operations must ' + \
                          'have identical sets of keys.')
