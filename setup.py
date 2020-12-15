import json
import os
from setuptools import find_namespace_packages, setup


_PACKAGE_NAMESPACE_NAME = 'arimo'

_METADATA_FILE_NAME = 'metadata.json'

_SETUP_REQUIREMENTS_FILE_NAME = 'requirements-setup.txt'
_INSTALL_REQUIREMENTS_FILE_NAME = 'requirements.txt'


_metadata = json.load(open(_METADATA_FILE_NAME))


def parse_requirements(requirements_file_name):
    return [s for s in
                {i.strip()
                 for i in open(requirements_file_name).readlines()}
            if not s.startswith('#')]


setup(
    name=_metadata['PACKAGE'],
    version=_metadata['VERSION'],
    namespace_packages=[_PACKAGE_NAMESPACE_NAME],
    packages=find_namespace_packages(include=[f'{_PACKAGE_NAMESPACE_NAME}.*']),
    setup_requires=parse_requirements(_SETUP_REQUIREMENTS_FILE_NAME),
    install_requires=parse_requirements(_INSTALL_REQUIREMENTS_FILE_NAME))
