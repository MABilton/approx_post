import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class, tree_flatten, tree_unflatten
from .base import ArrayContainer

@register_pytree_node_class
class JaxContainer(ArrayContainer):

    _ATTR_TO_SEARCH = ('linalg', 'fft')

    # May want only floats stored for autograd purposes:    
    def __init__(self, contents, floats_only=False, convert_inputs=True):
        super().__init__(contents)
        if convert_inputs:
          self._convert_contents_to_jax(floats_only, convert_inputs)

    def _convert_contents_to_jax(self, floats_only, convert_inputs):
        # Check that everything can be converted to Jax Numpy array:
        for i, key_i in enumerate(self.keys()):
            element_i = self.contents[key_i]
            # Try convert element_i to jax.numpy array if requested:
            try:
                element_i = jnp.array(element_i)
                if floats_only:
                    element_i = element_i.astype(float)
            except TypeError:
                raise TypeError(f"Element {i} of type {type(element_i)} " + \
                                 "cannot be converted to jax.numpy array.")
            self.contents[key_i] = element_i
      
    def _manage_function_call(self, func, types, *args, **kwargs):

      output_dict = {}

      # Identify all instances of containers in args:
      self._check_container_compatability(args, kwargs)

      # Next, need to 'find' Jax implementation of function:
      try:
        # First try to access function directly in jax.numpy:
        jax_method =  getattr(jnp, str(func.__name__))
      except AttributeError:
        # If that doesn't work, try search jax.numpy.linalg and jax.numpy.fft:
        for i, attr in enumerate(self._ATTR_TO_SEARCH):
          try:
            jax_method =  getattr(getattr(jnp, attr), str(func.__name__))
            break
          except AttributeError:
            if i == len(self._ATTR_TO_SEARCH)-1:
              raise AttributeError(f'The {func.__name__} method is not implemented in jax.numpy.')

      for key in self.keys():
        args_i = self._prepare_args(args, key)
        kwargs_i = self._prepare_kwargs(kwargs, key)
        output_dict[key] = jax_method(*args_i, **kwargs_i)

      if self._type is list:
        output_list = list(output_dict.values())
        output_container =  JaxContainer(output_list)
      else:
        output_container =  JaxContainer(output_dict)

      return output_container
      
    # Functions required by @register_pytree_node_class decorator:
    def tree_flatten(self):
      return tree_flatten(self.contents)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      return cls(tree_unflatten(aux_data, children), convert_inputs=False)

    def _set_array_item(self, container_key, idx, value_i):
      self.contents[container_key] = self.contents[container_key].at[idx].set(value_i)

    def array(self, in_array):
      return jnp.array(in_array)