from jax.tree_util import register_pytree_node_class, tree_flatten, tree_unflatten

@register_pytree_node_class
class JaxContainer(ArrayContainer):

    _ATTR_TO_SEARCH = ('linalg', 'fft')

    # May want only floats stored for autograd purposes:    
    def __init__(self, contents, floats_only=True, convert_inputs=True):
        super().__init__(contents)
        if convert_inputs:
          self.convert_contents_to_jax(floats_only, convert_inputs)

    def convert_contents_to_jax(self, floats_only, convert_inputs):
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
      

    def manage_function_call(self, func, types, *args, **kwargs):

      output_dict = {}

      # Identify all instances of containers in args:
      container_list, arg_list = self.find_containers_in_args(args)
      if container_list:
        self.check_container_compatability(container_list)

      for key in self.keys():
        contents_i = [cont[key] for cont in container_list]
        
        # First try to access function directly in jax.numpy:
        try:
          jax_method =  getattr(jnp, str(func.__name__))
          output_dict[key] = jax_method(*contents_i, *arg_list)
        except AttributeError:
          pass

        # If that doesn't work, try jax.numpy.linalg and jax.numpy.fft:
        for i, attr in enumerate(self._ATTR_TO_SEARCH):
          try:
            jax_method =  getattr(getattr(jnp, attr), str(func.__name__))
            output_dict[key] = jax_method(*contents_i, *arg_list)
            break
          except AttributeError:
            if i == len(self._ATTR_TO_SEARCH)-1:
              raise AttributeError(f'The {func.__name__} method is not implemented in jax.numpy.')

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