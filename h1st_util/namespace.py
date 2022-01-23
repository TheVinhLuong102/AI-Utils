"""Namespace utilities."""


from __future__ import annotations

import argparse
import copy
import datetime
import json
from types import ModuleType
from typing import Any, Optional, Union
from typing import List, Tuple   # Py3.9+: use built-ins


__all__ = ('Namespace',)


class Namespace(argparse.Namespace):
    """Namespace with support for nested keys."""

    @staticmethod
    def _as_namespace_if_applicable(obj: Any, /) -> Union[Namespace, Any]:
        """Try to create namespace from object."""
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
            obj: Namespace = Namespace(**obj)

        elif isinstance(obj, argparse.Namespace):
            obj: Namespace = Namespace(**obj.__dict__)

        elif isinstance(obj, ModuleType):
            # then get module's non-special, non-module-typed members only
            obj: Namespace = Namespace(**{
                k: v
                for k, v in obj.__dict__.items()
                if not (k.startswith('__') or isinstance(v, ModuleType))})

        return obj

    def __init__(self, **kwargs: Any):
        """Init Namespace."""
        self.__dict__['__metadata__'] = kwargs.pop('__metadata__', {})

        super().__init__(**{k: self._as_namespace_if_applicable(v)
                            for k, v in kwargs.items()})

        # pylint: disable=invalid-name
        for k, v in self.__metadata__.copy().items():
            nested_attr_names_list: List[str] = k.split(sep='.', maxsplit=-1)

            if len(nested_attr_names_list) > 1:
                del self.__metadata__[k]

                self._get_nested_attr(nested_attr_names_list[:-1]) \
                    .__metadata__[nested_attr_names_list[-1]] = v

    @staticmethod
    def pprint(namespace_or_dict: Union[Namespace, dict], /,
               *, indent: int = 0, addl_indent: int = 2):
        # pylint: disable=too-many-locals
        """Pretty-print namespace or dict."""
        indent_str: str = indent * ' '
        single_addl_indent_str: str = (indent + addl_indent) * ' '
        double_addl_indent_str: str = (indent + 2 * addl_indent) * ' '

        s: str = indent_str + '{'   # pylint: disable=invalid-name

        d: dict = (namespace_or_dict.__dict__   # pylint: disable=invalid-name
                   if isinstance(namespace_or_dict, Namespace)
                   else namespace_or_dict)

        if d:   # pylint: disable=too-many-nested-blocks
            s += '\n'   # pylint: disable=invalid-name

            for k, v in d.items():   # pylint: disable=invalid-name
                if k != '__metadata__':
                    v_metadata_str: str = ''

                    if isinstance(namespace_or_dict, Namespace):
                        v_metadata: Union[argparse.Namespace, dict] = \
                            namespace_or_dict.__metadata__.get(k)

                        if v_metadata:
                            if isinstance(v_metadata, argparse.Namespace):
                                v_metadata: dict = v_metadata.__dict__

                            label: Optional[str] = v_metadata.get('label')
                            if label:
                                v_metadata_str += (
                                    double_addl_indent_str +
                                    f'{label}\n'
                                )

                            description: Optional[str] = \
                                v_metadata.get('description')
                            if description:
                                v_metadata_str += (
                                    double_addl_indent_str +
                                    f'({description})\n'
                                )

                            choices: Optional[Any] = v_metadata.get('choices')
                            if choices:
                                v_metadata_str += (
                                    double_addl_indent_str +
                                    'choices:\n' +
                                    '\n'.join((double_addl_indent_str +
                                               f'    - {choice}')
                                              for choice in choices) +
                                    '\n'
                                )

                            default: Optional[Any] = v_metadata.get('default')
                            if default:
                                v_metadata_str += (
                                    double_addl_indent_str +
                                    f'default:   {default}\n'
                                )

                            tags: Optional[str] = v_metadata.get('tags')
                            if tags:
                                v_metadata_str += (
                                    double_addl_indent_str +
                                    'tags:   ' +
                                    ', '.join(tags) +
                                    '\n'
                                )

                    s += (   # pylint: disable=invalid-name
                        single_addl_indent_str +
                        f'{k} = '
                    ) + (
                        ('\n' + v_metadata_str +
                         Namespace.pprint(v, indent=indent + 2 * addl_indent))
                        if isinstance(v, (Namespace, dict))
                        else (
                            f'{v}\n' +
                            (f'{v_metadata_str}\n' if v_metadata_str else '')
                        )
                    )

        s += (indent_str + '}\n\n')   # pylint: disable=invalid-name

        return s

    def __repr__(self) -> str:
        """Return string repr."""
        return self.pprint(self)

    def __str__(self) -> str:
        """Return string repr."""
        return repr(self)

    @classmethod
    def create(cls, obj: Any) -> Union[Namespace, Any]:
        """(Try to) Create namespace from object."""
        return cls._as_namespace_if_applicable(obj)

    def _get_nested_attr(self, nested_attr_names: List[str], /) -> Any:
        nested_attr = self

        for nested_attr_name in nested_attr_names:
            nested_attr = nested_attr.__getattribute__(nested_attr_name)

        return nested_attr

    def __getattr__(self, attr: str, /) -> Any:
        """Get (nested) attribute value by (nested) attribute name."""
        return self._get_nested_attr(attr.split(sep='.', maxsplit=-1))

    def __getitem__(self, item: str, /) -> Any:
        """Get (nested) item value by (nested) item name."""
        return getattr(self, item)

    def __setattr__(self, attr: str, value: Any, /):
        """Set (nested) attribute value by (nested) attribute name."""
        nested_attr_names: List[str] = attr.split(sep='.', maxsplit=-1)

        value = self._as_namespace_if_applicable(value)

        if len(nested_attr_names) > 1:
            nested_ns: Namespace = self._get_nested_attr(nested_attr_names[:-1])   # noqa: E501
            setattr(nested_ns, nested_attr_names[-1], value)

        else:
            self.__dict__[attr] = value

    def __setitem__(self, item: str, value: Any, /):
        """Set (nested) item value by (nested) item name."""
        setattr(self, item, value)

    def __delattr__(self, attr: str, /):
        """Delete (nested) attr."""
        nested_attr_names: List[str] = attr.split(sep='.', maxsplit=-1)

        del self._get_nested_attr(nested_attr_names[:-1]) \
                .__dict__[nested_attr_names[-1]]

    def __delitem__(self, item: str, /):
        """Delete (nested) item."""
        delattr(self, item)

    def __iter__(self):
        """Iterate through content."""
        return (k for k in self.__dict__ if k != '__metadata__')

    def update(self,   # noqa: MC0001
               other: Union[argparse.Namespace, dict, ModuleType] = {}, /,
               **kwargs: Any):
        # pylint: disable=dangerous-default-value,too-many-branches
        """Update content."""
        if isinstance(other, argparse.Namespace):
            other = copy.deepcopy(other.__dict__)

        elif isinstance(other, dict):
            other = copy.deepcopy(other)

        elif isinstance(other, ModuleType):
            other = {k: v
                     for k, v in other.__dict__.items()
                     if not (k.startswith('__') or isinstance(v, ModuleType))}

        else:
            raise ValueError('*** `other` must be Namespace, Dict or Module ***')   # noqa: E501

        __modules_first__ = kwargs.pop('__modules_first__', False)
        other.update(copy.deepcopy(kwargs))

        __metadata__ = other.get('__metadata__', {})

        if __modules_first__:
            for k, v in other.items():   # pylint: disable=invalid-name
                if k != '__metadata__':
                    n = getattr(self, k, None)   # pylint: disable=invalid-name
                    if isinstance(n, Namespace) and isinstance(v, ModuleType):
                        n.update(v, __modules_first__=True)

            for k, v in other.items():   # pylint: disable=invalid-name
                if k != '__metadata__':
                    n = getattr(self, k, None)   # pylint: disable=invalid-name
                    if isinstance(n, Namespace) and \
                            isinstance(v, (dict, argparse.Namespace)):
                        n.update(v, __modules_first__=True)
                    elif not isinstance(v, ModuleType):
                        setattr(self, k, v)

        else:
            for k, v in other.items():   # pylint: disable=invalid-name
                if k != '__metadata__':
                    n = getattr(self, k, None)   # pylint: disable=invalid-name
                    if isinstance(n, Namespace) and \
                            isinstance(v, (dict, argparse.Namespace, ModuleType)):   # noqa: E501
                        n.update(v)
                    else:
                        setattr(self, k, v)

        for k, v in __metadata__.items():   # pylint: disable=invalid-name
            nested_attr_names: List[str] = k.split(sep='.', maxsplit=-1)
            self._get_nested_attr(nested_attr_names[:-1]) \
                .__metadata__[nested_attr_names[-1]] = v

    def keys(self, all_nested: bool = False) -> List[str]:
        """Get (nested) keys."""
        if all_nested:
            keys: List[str] = []

            for k, v in self.__dict__.items():   # pylint: disable=invalid-name
                if k != '__metadata__':
                    keys.append(k)

                    if isinstance(v, Namespace):
                        keys.extend(f'{k}.{sub_k}'
                                    for sub_k in v.keys(all_nested=True))

            return keys

        return [k for k in self.__dict__ if k != '__metadata__']

    def values(self) -> List[Any]:
        """Get values."""
        return [v for k, v in self.__dict__.items() if k != '__metadata__']

    def items(self) -> List[Tuple[str, Any]]:
        """Get items."""
        return [i for i in self.__dict__.items() if i[0] != '__metadata__']

    def get(self, key: str, default: Optional[Any] = None):
        """Get item by key string, with a default fall-back value."""
        return self.__dict__.get(key, default)

    def __len__(self):
        """Count number of items."""
        return len(self.keys())

    def __call__(self, key: str, /):
        """Get metadata of a certain key."""
        nested_attr_names: List[str] = key.split(sep='.', maxsplit=-1)

        return (self._get_nested_attr(nested_attr_names[:-1])
                .__metadata__.get(nested_attr_names[-1], Namespace()))

    def to_dict(self):
        """Convert to Dict."""
        def _dict_no_inf(d: dict, /) -> dict:   # pylint: disable=invalid-name
            return {k: (_dict_no_inf(v)
                        if isinstance(v, dict)
                        else (None if str(v)[-3:] == 'inf' else v))
                    for k, v in d.items()}

        return {k: (v.to_dict()
                    if isinstance(v, Namespace)
                    else (_dict_no_inf(v)
                          if isinstance(v, dict)
                          else (None if str(v)[-3:] == 'inf' else v)))
                for k, v in self.items()}

    class _JSONEncoder(json.JSONEncoder):
        def default(self, obj):   # pylint: disable=arguments-renamed
            def _serializable(x: Any):   # pylint: disable=invalid-name
                return ([_serializable(i) for i in x]
                        if isinstance(x, (list, set, tuple))
                        else ({k: _serializable(v)
                               for k, v in x.items()}
                              if isinstance(x, (dict, Namespace))
                              else (str(x)
                                    if isinstance(x, (datetime.datetime,
                                                      datetime.time))
                                    else (None if str(x)[-3:] == 'inf' else x))))   # noqa: E501

            if isinstance(obj, (list, set, tuple, dict, Namespace)):
                return _serializable(obj)

            return json.JSONEncoder.default(self, obj)

    def to_json(self, path: str):
        """Dump content to JSON file."""
        with open(file=path,
                  mode='wt',
                  buffering=-1,
                  encoding='utf-8',
                  errors='strict',
                  newline=None,
                  closefd=True,
                  opener=None) as json_file:
            json.dump(obj=self,
                      fp=json_file,
                      skipkeys=False,
                      ensure_ascii=False,
                      check_circular=True,
                      allow_nan=True,
                      cls=self._JSONEncoder,
                      indent=2,
                      separators=None,
                      default=None,
                      sort_keys=False)
