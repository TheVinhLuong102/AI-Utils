import inspect

import six

from . import iterables


_SELF_ONLY_METHOD_ARG_SPEC = \
    inspect.ArgSpec(
        args=['self'],
        varargs=None,
        keywords=None,
        defaults=None)


def _arg_spec(obj):   # for Py2 compatibility, using inspect.getargspec instead of .getfullargspec / .signature
    try:
        return inspect.getargspec(func=obj)
    except TypeError:
        pass


def enable_inplace(Class):
    def enable_inplace(method):
        def method_with_inplace(*args, **kwargs):
            """
            inplace (bool): whether to update the instance in-place;
                *IMPORTANT NOTE 1:* please specify all other arguments by **keyword arguments**
                *IMPORTANT NOTE 2:* inplace=True cannot be used with varargs
            """
            # try to retrieve __self__ instance from *args, falling back to method.__self__
            self = args[0] if args else method.__self__   # requires carrying __self__ along in all nested decorators

            inplace = kwargs.pop('inplace', False)
            kwargs_set = set(kwargs)

            method_arg_spec = inspect.getargspec(method)
            method_args = kwargs \
                if method_arg_spec.keywords \
                else {k: kwargs[k] for k in kwargs_set.intersection(method_arg_spec.args)}

            result = method(*args, **method_args)

            if inplace:
                inplace_kwargs_spec = inspect.getargspec(Class._inplace)
                inplace_kwargs = kwargs \
                    if inplace_kwargs_spec.keywords \
                    else {k: kwargs[k] for k in kwargs_set.intersection(inplace_kwargs_spec.args)}
                Class._inplace(self, result, **inplace_kwargs)

            elif callable(result) \
                    and (not isinstance(result, Class)) \
                    and (not inspect.isclass(result)) \
                    and (_arg_spec(result) != _SELF_ONLY_METHOD_ARG_SPEC):
                result.__self__ = self
                return enable_inplace(result)

            else:
                return result

        method_with_inplace.__module__ = method.__module__
        method_with_inplace.__name__ = method.__name__
        method_with_inplace.__doc__ = method.__doc__ + method_with_inplace.__doc__ \
            if method.__doc__ else ''

        if six.PY2:
            method_with_inplace.__self__ = method.__self__   # requires carrying __self__ along in all nested decorators

        return method_with_inplace

    if '_INPLACE_ABLE' in dir(Class):
        Class._INPLACE_ABLE = iterables.to_iterable(Class._INPLACE_ABLE)

        for method_name, method in \
                (inspect.getmembers(Class, predicate=inspect.ismethod)
                    # Py2: inspect.ismethod(...) covers both Bound & Unbound methods
                    # Py3: inspect.ismethod(...) covers only Bound methods
                 if six.PY2
                 else inspect.getmembers(Class)):
            if method_name in Class._INPLACE_ABLE:
                setattr(Class, method_name, enable_inplace(method))

    else:
        Class._INPLACE_ABLE = []

        for method_name, method in \
                (inspect.getmembers(Class, predicate=inspect.ismethod)
                    # Py2: inspect.ismethod(...) covers both Bound & Unbound methods
                    # Py3: inspect.ismethod(...) covers only Bound methods
                 if six.PY2
                 else inspect.getmembers(Class)):
            if (method_name in ('__call__', '__getattr__') or not method_name.startswith('__')) and \
                    (method_name != '_inplace') and (not method.__self__):
                setattr(Class, method_name, enable_inplace(method))
                Class._INPLACE_ABLE.append(method_name)

    return Class


def _docstr_settable_property(property):
    def f():
        """
        (**Assignable**/**Settable** **Property**)
        """
    property.__doc__ = \
        f.__doc__ + \
        ('' if property.__doc__ is None
            else property.__doc__)
    return property


def _docstr_verbose(method):
    def f():
        """
            - **verbose** (bool, or int 0-2): level of logging detail
                - ``0``/``False``: no logging
                - ``1``/``True``: log info messages from the current method call only
                - ``2``: log info messages from current method and all sub-method calls
        """
    method.__doc__ = \
        ('' if method.__doc__ is None
         else method.__doc__) + \
        f.__doc__
    return method


def _docstr_experimental(method):
    def f():
        """
        (**EXPERIMENTAL**)
        """
    method.__doc__ = \
        f.__doc__ + \
        ('' if method.__doc__ is None
            else method.__doc__)
    return method


def _docstr_for_back_compat(method):
    def f():
        """
        (**FOR BACKWARD COMPATIBILITY**)
        """
    method.__doc__ = \
        f.__doc__ + \
        ('' if method.__doc__ is None
            else method.__doc__)
    return method


class _DocStr_Deprecated:
    def __init__(self, since_ver=None, replacement=None):
        self.since_ver = since_ver
        self.replacement = replacement

    def __call__(self, method):
        def f():
            """
            (**DEPRECATED** since version %s%s)
            """
        method.__doc__ = \
            f.__doc__ % (self.since_ver, ', replaced by ``%s``' % self.replacement if self.replacement else '') + \
            ('' if method.__doc__ is None
                else method.__doc__)
        return method
