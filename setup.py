from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Uncertainty Quantification for Computed Tomography'
LONG_DESCRIPTION = ''

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="uqct",
    version=VERSION,
    author="Matteo Gätzner, Johannes Kirschner",
    author_email="<youremail@email.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
)