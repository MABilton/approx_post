import numpy as np
from .array import ArrayContainer

class NumpyContainer(ArrayContainer):

    def __init__(self, contents):
        super().__init__(contents)
        self._convert_contents_to_numpy()

    def _convert_contents_to_numpy(self):
        for i, key_i in enumerate(self.keys()):
            contents_i = self.contents[key_i]
            try:
                self.contents[key_i] = np.array(contents_i)
            except TypeError:
                error_msg = f"""Element {i} of type {type(contents_i)} 
                                cannot be converted to numpy array."""
                raise TypeError(error_msg)
                
    def _manage_function_call(self, func, types, *args, **kwargs):

      output_dict = {}
      
      # Check containers are compatable in operation:
      self._check_container_compatability(args, kwargs)

      for key in self.keys():
        args_i = self._prepare_args(args, key)
        kwargs_i = self._prepare_kwargs(kwargs, key)
        output_dict[key] = func(*args_i, **kwargs_i)
      
      if self._type is list:
        output_list = list(output_dict.values())
        output_container = NumpyContainer(output_list)
      else:
          output_container = NumpyContainer(output_dict)

      return output_container
  
    def _set_array_item(self, container_key, idx, value_i):
      self.contents[container_key][self.array(idx)] = value_i
    
    def array(self, in_array):
      return np.array(in_array)