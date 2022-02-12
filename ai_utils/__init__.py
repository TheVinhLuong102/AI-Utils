"""H1st utilities package metadata."""


from importlib.metadata import version


__all__ = ('__version__',)


__version__: str = version(distribution_name='AI-Utils')
