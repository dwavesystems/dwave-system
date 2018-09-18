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


install_requires = ['dimod>=0.7.0,<0.8.0',
                    'dwave-cloud-client>=0.4.13,<0.5.0',
                    'dwave-networkx>=0.6.0,<0.7.0',
                    'homebase>=1.0.0,<2.0.0',
                    'minorminer>=0.1.3,<0.2.0',
                    'six>=1.11.0,<2.0.0',
                    'numpy>=1.14.0,<2.0.0',
                    ]

extras_require = {'drivers': ['dwave-drivers>=0.4.0,<0.5.0']}

packages = ['dwave',
            'dwave.system',
            'dwave.system.cache',
            'dwave.system.composites',
            'dwave.system.samplers',
            ]

setup(
    name='dwave-system',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    url='https://github.com/dwavesystems/dwave-system',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False
)
