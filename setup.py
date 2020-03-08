import os
from ruamel import yaml
from setuptools import find_namespace_packages, find_packages, setup


_PACKAGE_NAMESPACE_NAME = 'arimo'

_METADATA_FILE_NAME = 'metadata.yml'

_SETUP_REQUIREMENTS_FILE_NAME = 'requirements-setup.txt'

_INSTALL_REQUIREMENTS_FILE_NAME = 'requirements.txt'


_metadata = \
    yaml.safe_load(
        stream=open(os.path.join(
                os.path.dirname(__file__),
                _PACKAGE_NAMESPACE_NAME,
                _METADATA_FILE_NAME)))


setup(
    name=_metadata['PACKAGE'],
    author=_metadata['AUTHOR'],
    author_email=_metadata['AUTHOR_EMAIL'],
    url=_metadata['URL'],
    version=_metadata['VERSION'],
    description=_metadata['DESCRIPTION'],
    long_description=_metadata['DESCRIPTION'],
    keywords=_metadata['DESCRIPTION'],
    namespace_packages=[_PACKAGE_NAMESPACE_NAME],
    packages=find_namespace_packages(include=[f'{_PACKAGE_NAMESPACE_NAME}.*']),
    include_package_data=True,
    setup_requires=
        [s for s in
            {i.strip()
             for i in open(_SETUP_REQUIREMENTS_FILE_NAME).readlines()}
         if not s.startswith('#')],
    install_requires=
        [s for s in
            {i.strip()
             for i in open(_INSTALL_REQUIREMENTS_FILE_NAME).readlines()}
         if not s.startswith('#')])
