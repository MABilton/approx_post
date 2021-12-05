import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class, tree_flatten, tree_unflatten
from .array import ArrayContainer

@register_pytree_node_class
class JaxContainer(ArrayContainer):

    # May want only floats stored for autograd purposes:    
    def __init__(self, contents, convert_inputs=True, floats_only=False):
        super().__init__(contents, convert_inputs)
        if convert_inputs:
          self._convert_contents_to_jax(floats_only)

    def _convert_contents_to_jax(self, floats_only):
        # Check that everything can be converted to Jax Numpy array:
        for i, key_i in enumerate(self.keys()):
            element_i = self.contents[key_i]
            if not issubclass(type(element_i), ArrayContainer):
              # Try convert element_i to jax.numpy array if requested:
              try:
                  element_i = jnp.array(element_i)
                  if floats_only:
                      element_i = element_i.astype(float)
              except TypeError:
                  error_msg = f"""Element {i} of type {type(element_i)} 
                                  cannot be converted to jax.numpy array."""
                  raise TypeError(error_msg)
              self.contents[key_i] = element_i
      
    def _manage_function_call(self, func, types, *args, **kwargs):

      output_dict = {}

      self._check_container_compatability(args, kwargs)

      for key in self.keys():
        args_i = self._prepare_args(args, key)
        kwargs_i = self._prepare_kwargs(kwargs, key)
        
        # Check to see if args or kwargs contains a container type:
        includes_containers = self._find_containers(args_i, kwargs_i)
        # If function call does not include containers, we need to remove any 'out' kwargs:
        method = self._find_method(jnp, func) if not includes_containers else self._find_method(np, func)
        output_dict[key] = method(*args_i, **kwargs_i)

      if self._type is list:
        output_list = list(output_dict.values())
        output_container = JaxContainer(output_list)
      else:
        output_container = JaxContainer(output_dict)

      return output_container
    
    # Removes 'out' from kwargs - not use by Jax methods:
    def _prepare_kwargs(self, kwargs, key):
      kwargs = super()._prepare_kwargs(kwargs, key)
      kwargs.pop('out', None)
      return kwargs

    def _find_containers(self, args, kwargs):
      containers_in_args = [issubclass(type(arg_i), ArrayContainer) for arg_i in args]
      containers_in_kwargs = [issubclass(type(arg_i), ArrayContainer) for arg_i in kwargs.values()]
      includes_containers = any(containers_in_args) or any(containers_in_kwargs)
      return includes_containers

    def _find_method(self, module, func, submodule_to_search=('', 'linalg', 'fft')):
      method_name = str(func.__name__)
      modules_to_search = [getattr(module, submodule, module) for submodule in submodule_to_search]
      for i, mod in enumerate(modules_to_search):
        try:
          found_method = getattr(mod, method_name)
          break
        except AttributeError:
          if i == len(submodule_to_search)-1:
            error_msg = f'The {method_name} method is not implemented in {module}.'
            raise AttributeError(error_msg)
      return found_method

    # Functions required by @register_pytree_node_class decorator:
    def tree_flatten(self):
      return tree_flatten(self.unpacked)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      try:
        unflattened = cls(tree_unflatten(aux_data, children), convert_inputs=True)
      except TypeError:
        unflattened = cls(tree_unflatten(aux_data, children), convert_inputs=False)
      return unflattened

    def _set_array_item(self, container_key, idx, value_i):
      self.contents[container_key] = self.contents[container_key].at[idx].set(value_i)

    def array(self, in_array):
      return jnp.array(in_array)