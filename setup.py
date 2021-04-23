import json
import os
from setuptools import find_namespace_packages, setup


_PACKAGE_NAMESPACE_NAME = 'h1st'

_METADATA_FILE_NAME = 'metadata.json'


_metadata = json.load(open(_METADATA_FILE_NAME))


setup(
    name=_metadata['PACKAGE'],
    version=_metadata['VERSION'],
    namespace_packages=[_PACKAGE_NAMESPACE_NAME, 'arimo'],
    packages=find_namespace_packages(include=[f'{_PACKAGE_NAMESPACE_NAME}.*',
                                              'arimo.*']),
    install_requires=open('requirements.txt').readlines())
