from setuptools import find_namespace_packages, setup


setup(
    name='Arimo-SDK-dev',
    author='Arimo LLC (a Panasonic company)',
    author_email='DSAR@Arimo.com',
    url='https://github.com/adatao/SDK-dev',
    version='0.0.0',
    description='Arimo Software Development Kit (SDK)',
    long_description='Arimo Software Development Kit (SDK)',
    keywords='Arimo Software Development Kit (SDK)',
    packages=find_namespace_packages(include=['arimo.*']))
