import json
from json import JSONDecodeError
import os

class Distribution:

    def __init__(self, func_dict, attr_dict):
        self._func_dict = func_dict
        self._attr_to_save = attr_dict

    def __repr__(self):
        try:
            repr_str = f'{self.__class__.__name__} object with parameters {repr(self.phi)}'
        except AttributeError:
            repr_str = f'{self.__class__.__name__} object with no specified parameters'
        return repr_str