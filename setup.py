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
    execfile(os.path.join(".", "dwave", "system", "package_info", "package_info.py"))
else:
    exec(open(os.path.join(".", "dwave", "system", "package_info", "package_info.py")).read())

install_requires = ['dimod>=0.6.1,<0.7.0',
                    'dwave_cloud_client>=0.3.1,<0.5.0',
                    'dwave-embedding-utilities>=0.2.0,<0.3.0',
                    'dwave-networkx>=0.6.0,<0.7.0',
                    'homebase>=1.0.0,<2.0.0',
                    'minorminer>=0.1.3,<0.2.0',
                    'six>=1.11.0,<2.0.0',
                    'dwave-system-tuning>=0.1.1,<0.2.0']

extras_require = {}

packages = ['dwave',
            'dwave.system',
            'dwave.system.samplers',
            'dwave.system.composites',
            'dwave.system.cache',
            'dwave.system.embedding',
            'dwave.system.exceptions',
            'dwave.system.flux_bias_offsets',
            'dwave.system.package_info']

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
    zip_safe=False
)
