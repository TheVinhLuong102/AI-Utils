from __future__ import print_function

import abc
import six


class _ABC(object):   # syntax "(metaclass=abc.ABCMeta)" only available in Py3
    __metaclass__ = abc.ABCMeta

    @classmethod
    @abc.abstractproperty
    def ab_cls_prp_2(self):
        raise NotImplementedError

    if six.PY3:
        @property
        @abc.abstractclassmethod
        def ab_cls_prp_3(self):
            raise NotImplementedError

    @abc.abstractproperty
    def ab_prp_2(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ab_prp_3(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def ab_cls_mthd_2(cls, x):
        raise NotImplementedError

    if six.PY3:
        @abc.abstractclassmethod
        def ab_cls_mthd_3(cls, x):
            raise NotImplementedError

    @abc.abstractmethod
    def ab_mthd(self, x):
        raise NotImplementedError


class C(_ABC):
    def __init__(self, x=0):
        self.x = x

    ab_cls_prp_2 = 'ABSTRACT CLASS PROPERTY v2'

    ab_cls_prp_3 = 'ABSTRACT CLASS PROPERTY v3'

    @property
    def ab_prp_2(self):
        return 'ABSTRACT PROPERTY v2 = {}'.format(self.x)

    @property
    def ab_prp_3(self):
        return 'ABSTRACT PROPERTY v3 = {}'.format(self.x)

    @classmethod
    def ab_cls_mthd_2(cls, x=0):
        return 'ABSTRACT CLASS METHOD v2 = {}'.format(x)

    @classmethod
    def ab_cls_mthd_3(cls, x=0):
        return 'ABSTRACT CLASS METHOD v3 = {}'.format(x)

    def ab_mthd(self, x=0):
        return 'ABSTRACT METHOD = {} ** {} = {}'.format(self.x, x, self.x ** x)

    cls_prp = 'CLASS PROPERTY'

    @property
    def prp(self):
        return 'PROPERTY = {}'.format(self.x)

    @classmethod
    def cls_mthd(self, x=0):
        return 'CLASS METHOD = {}'.format(x)

    def mthd(self, x=0):
        return 'METHOD = {} ** {} = {}'.format(self.x, x, self.x ** x)


obj = C(3)

print(C.ab_cls_prp_2)

if six.PY3:
    print(C.ab_cls_prp_3)

print(obj.ab_prp_2)
print(obj.ab_prp_3)
print(C.ab_cls_mthd_2(3))

if six.PY3:
    print(C.ab_cls_mthd_3(3))

print(obj.ab_mthd(3))
print(C.cls_prp)
print(obj.prp)
print(C.cls_mthd(3))
print(obj.mthd(3))
