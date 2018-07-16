from setuptools import find_packages, setup


_REQUIREMENTS_FILE_NAME = 'requirements.txt'


setup(
    name='Arimo-BAI-dev',
    author='Arimo LLC (a Panasonic company)',
    author_email='info@arimo.com',
    url='https://github.com/adatao/BAI-dev',
    version='0.0.0',
    description='Arimo BAI dev',
    long_description='Arimo BAI dev',
    keywords='Arimo BAI dev',
    packages=find_packages(),
    include_package_data=True,
    install_requires=
        [s for s in
            {i.strip()
             for i in open(_REQUIREMENTS_FILE_NAME).readlines()}
         if not s.startswith('#')])
