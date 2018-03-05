from __future__ import absolute_import

import sys
import os
from setuptools import setup

# add __version__, __author__, __authoremail__, __description__ to this namespace
_PY2 = sys.version_info.major == 2

# change directories so this works when called from other locations. Useful in build systems.
setup_folder_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_folder_loc)

if _PY2:
    execfile(os.path.join(".", "dwave", "system", "package_info.py"))
else:
    exec(open(os.path.join(".", "dwave", "system", "package_info.py")).read())

install_requires = ['dimod==0.6.0.dev2',
                    'six>=1.11.0,<2.0.0']

extras_require = {}

packages = ['dwave.system', 'dwave.system.samplers', 'dwave.system.composites', 'dwave.system.cache']

setup(
    name='dwave_system',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    url='https://github.com/dwavesystems/dwave_system',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    namespace_packages=['dwave']
)
