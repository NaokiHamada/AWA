# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('awa/_metadata.py') as f:
    # Work-around for importing metadata without importing external modules
    # that this package depends on.
    # In spirit, this is equivalent to:
    # from awa._metadata import (__package_name__, __version__, __author__,
    #                            __email__, __license__, __url__,
    #                            __description__, __long_description__)
    exec(f.read())

with open('requirements.txt') as f:
    # Read requirements
    requirements = f.read().split()

setup(
    name=__package_name__,
    version=__version__,
    author=__author__,
    author_email=__email__,
    license=__license__,
    url=__url__,
    description=__description__,
    long_description=__long_description__,
    packages=find_packages(),
    install_requires=requirements)
