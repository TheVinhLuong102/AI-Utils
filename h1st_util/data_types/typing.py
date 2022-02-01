"""Data Typing."""


from typing import Union   # pylint: disable=import-self


__all__ = (
    'PyNumType',
    'PyPossibleFeatureType',
)


PyNumType = Union[float, int]
PyPossibleFeatureType = Union[bool, float, int, str]
