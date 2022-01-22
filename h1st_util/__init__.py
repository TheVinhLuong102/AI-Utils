"""H1st Util package."""


import argparse
import copy
import datetime
import importlib
from importlib.metadata import version
import json
import os
import re
import tempfile
from types import ModuleType
from typing import Sequence   # TODO: Py3.9: use generics/collections.abc


__all__: Sequence[str] = ('__version__',)


__version__: str = version(distribution_name='H1st-Util')


class DefaultDict(dict):
    def __init__(self, default, *args, **kwargs):
        super(DefaultDict, self).__init__(*args, **kwargs)
        self._default = \
            default \
            if callable(default) \
            else lambda: default

    def __getitem__(self, item):
        return super(DefaultDict, self).__getitem__(item) \
            if item in self \
          else self._default()

    @property
    def default(self):
        return self._default()

    @default.setter
    def default(self, default):
        if callable(default):
            self._default = default
        elif default != self._default():
            self._default = lambda: default


def import_obj(name):
    module_name, obj_name = name.rsplit('.', 1)
    return getattr(
        importlib.import_module(module_name),
        obj_name)


class Namespace(argparse.Namespace):
    @staticmethod
    def _as_namespace_if_applicable(obj):
        if isinstance(obj, dict):
            keys_all_str = True

            for k in obj:
                if isinstance(k, str):
                    try:
                        str(k)
                    except:
                        keys_all_str = False

                else:
                    keys_all_str = False

            if keys_all_str:
                obj = Namespace(**obj)

        elif isinstance(obj, argparse.Namespace):
            obj = Namespace(**obj.__dict__)

        elif isinstance(obj, ModuleType):   # then get module's non-special, non-module-typed members only
            obj = Namespace(
                **{k: v
                   for k, v in obj.__dict__.items()
                   if not (k.startswith('__') or isinstance(v, ModuleType))})

        return obj

    def __init__(self, **kwargs):
        self.__dict__['__metadata__'] = \
            kwargs.pop('__metadata__', {})

        super(Namespace, self).__init__(
            **{k: self._as_namespace_if_applicable(v)
               for k, v in kwargs.items()})

        for k, v in self.__metadata__.copy().items():
            nested_attr_names_list = k.split('.')
            if len(nested_attr_names_list) > 1:
                del self.__metadata__[k]
                self._get_nested_attr(nested_attr_names_list[:-1]) \
                    .__metadata__[nested_attr_names_list[-1]] = v

    def __repr__(self):
        def pprint(namespace_or_dict, indent=0, addl_indent=2):
            indent_str = indent * ' '
            single_addl_indent_str = (indent + addl_indent) * ' '
            double_addl_indent_str = (indent + 2 * addl_indent) * ' '

            s = indent_str + '{'

            d = namespace_or_dict.__dict__ \
                if isinstance(namespace_or_dict, Namespace) \
                else namespace_or_dict

            if d:
                s += '\n'

                for k, v in d.items():
                    if k != '__metadata__':
                        v_metadata_str = ''

                        if isinstance(namespace_or_dict, Namespace):
                            v_metadata = namespace_or_dict.__metadata__.get(k)

                            if v_metadata:
                                if isinstance(v_metadata, argparse.Namespace):
                                    v_metadata = v_metadata.__dict__

                                label = v_metadata.get('label')
                                if label:
                                    v_metadata_str += \
                                        (double_addl_indent_str + label + '\n')

                                description = v_metadata.get('description')
                                if description:
                                    v_metadata_str += \
                                        (double_addl_indent_str +
                                         '(' + description + ')\n')

                                choices = v_metadata.get('choices')
                                if choices:
                                    v_metadata_str += \
                                        (double_addl_indent_str +
                                         'choices:\n' +
                                         '\n'.join(
                                            (double_addl_indent_str + '    - ' + str(choice))
                                            for choice in choices) +
                                         '\n')

                                if 'default' in v_metadata:
                                    v_metadata_str += \
                                        (double_addl_indent_str +
                                         'default:   ' + str(v_metadata['default']) + '\n')

                                tags = v_metadata.get('tags')
                                if tags:
                                    v_metadata_str += \
                                        (double_addl_indent_str +
                                         'tags:   ' + ', '.join(tags) + '\n')

                        s += (single_addl_indent_str +
                              '{} = '.format(k)) + \
                            (('\n' +
                              v_metadata_str +
                              pprint(v, indent=indent + 2 * addl_indent))
                             if isinstance(v, (Namespace, dict))
                             else ('{}\n'.format(v) +
                                   ((v_metadata_str + '\n')
                                    if v_metadata_str
                                    else '')))

            s += (indent_str + '}\n\n')

            return s

        return pprint(self)

    def __str__(self):
        return repr(self)

    @classmethod
    def create(cls, obj):
        return cls._as_namespace_if_applicable(obj)

    def _get_nested_attr(self, nested_attr_names_list):
        nested_attr = self
        for nested_attr_name in nested_attr_names_list:
            nested_attr = nested_attr.__getattribute__(nested_attr_name)
        return nested_attr

    def __getattr__(self, item):
        return self._get_nested_attr(item.split('.'))

    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setattr__(self, key, value):
        nested_attr_names_list = key.split('.')
        value = self._as_namespace_if_applicable(value)
        if len(nested_attr_names_list) > 1:
            nested_attr = self._get_nested_attr(nested_attr_names_list[:-1])
            setattr(nested_attr, nested_attr_names_list[-1], value)
        else:
            self.__dict__[key] = value

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delattr__(self, item):
        nested_attr_names_list = item.split('.')
        del self._get_nested_attr(nested_attr_names_list[:-1]) \
            .__dict__[nested_attr_names_list[-1]]

    def __delitem__(self, key):
        delattr(self, key)

    def __iter__(self):
        return (k for k in self.__dict__
                  if k != '__metadata__')

    def update(self, __other__={}, **kwargs):
        if isinstance(__other__, argparse.Namespace):
            __other__ = copy.deepcopy(__other__.__dict__)

        elif isinstance(__other__, dict):
            __other__ = copy.deepcopy(__other__)

        elif isinstance(__other__, ModuleType):
            __other__ = \
                {k: v
                 for k, v in __other__.__dict__.items()
                 if not (k.startswith('__') or isinstance(v, ModuleType))}

        else:
            raise ValueError('*** __other__ arg must be a Namespace, Dict or Module ***')

        __modules_first__ = kwargs.pop('__modules_first__', False)
        __other__.update(copy.deepcopy(kwargs))

        __metadata__ = __other__.get('__metadata__', {})

        if __modules_first__:
            for k, v in __other__.items():
                if k != '__metadata__':
                    n = getattr(self, k, None)
                    if isinstance(n, Namespace) and isinstance(v, ModuleType):
                        n.update(v, __modules_first__=True)

            for k, v in __other__.items():
                if k != '__metadata__':
                    n = getattr(self, k, None)
                    if isinstance(n, Namespace) and isinstance(v, (dict, argparse.Namespace)):
                        n.update(v, __modules_first__=True)
                    elif not isinstance(v, ModuleType):
                        setattr(self, k, v)

        else:
            for k, v in __other__.items():
                if k != '__metadata__':
                    n = getattr(self, k, None)
                    if isinstance(n, Namespace) and isinstance(v, (dict, argparse.Namespace, ModuleType)):
                        n.update(v)
                    else:
                        setattr(self, k, v)

        for k, v in __metadata__.items():
            nested_attr_names_list = k.split('.')
            self._get_nested_attr(nested_attr_names_list[:-1]) \
                .__metadata__[nested_attr_names_list[-1]] = v

    def keys(self, all_nested=False):
        if all_nested:
            keys = []
            for k, v in self.__dict__.items():
                if k != '__metadata__':
                    keys.append(k)
                    if isinstance(v, Namespace):
                        keys += ['{}.{}'.format(k, sub_k) for sub_k in v.keys(all_nested=True)]
            return keys
        else:
            return [k for k in self.__dict__
                      if k != '__metadata__']

    def values(self):
        return [v for k, v in self.__dict__.items()
                  if k != '__metadata__']

    def items(self):
        return [i for i in self.__dict__.items()
                  if i[0] != '__metadata__']

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __len__(self):
        return len(self.keys())

    def __call__(self, item):   # get metadata of a certain key
        nested_attr_names_list = item.split('.')
        return self._get_nested_attr(nested_attr_names_list[:-1]) \
            .__metadata__.get(nested_attr_names_list[-1], Namespace())

    def to_dict(self):
        def _dict_no_inf(d):
            return {k: _dict_no_inf(v)
                        if isinstance(v, dict) \
                        else (None if str(v)[-3:] == 'inf'
                                   else v)
                    for k, v in d.items()}

        return {k: v.to_dict()
                    if isinstance(v, Namespace)
                    else (_dict_no_inf(v)
                          if isinstance(v, dict)
                          else (None if str(v)[-3:] == 'inf'
                                     else v))
                for k, v in self.items()}

    class _JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            def _serializable(x):
                return [_serializable(i) for i in x] \
                    if isinstance(x, (list, set, tuple)) \
                    else ({k: _serializable(v)
                           for k, v in x.items()}
                          if isinstance(x, (dict, Namespace))
                          else (str(x)
                                if isinstance(x, (datetime.datetime, datetime.time))
                                else (None if str(x)[-3:] == 'inf'
                                           else x)))

            if isinstance(obj, (list, set, tuple, dict, Namespace)):
                return _serializable(obj)
            
            return json.JSONEncoder.default(self, obj)

    def to_json(self, path):
        json.dump(
            self,
            open(path, 'w'),
            cls=self._JSONEncoder,
            encoding='utf-8',
            ensure_ascii=False,
            indent=2)
