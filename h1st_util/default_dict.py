"""Default Dict utilities."""


from typing import Any


__all__ = ('DefaultDict',)


class DefaultDict(dict):
    """Dict with Default Value."""

    def __init__(self, default: Any, *args: Any, **kwargs: Any):
        """Init Default Dict."""
        super().__init__(*args, **kwargs)

        self._default: callable = (default
                                   if callable(default)
                                   else (lambda: default))

    def __getitem__(self, item: str, /) -> Any:
        """Get item."""
        return super().__getitem__(item) if item in self else self._default()

    @property
    def default(self) -> Any:
        """Get default value."""
        return self._default()

    @default.setter
    def default(self, default: Any, /):
        """Set default value."""
        if callable(default):
            self._default = default

        elif default != self._default():
            self._default = lambda: default
