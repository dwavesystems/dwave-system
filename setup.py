# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
import os

from setuptools import setup


# change directories so this works when called from other locations. Useful in build systems.
setup_folder_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_folder_loc)

exec(open(os.path.join(".", "dwave", "system", "package_info.py")).read())


install_requires = ['dimod>=0.10.0,<0.11.0',
                    'dwave-cloud-client>=0.9.1,<0.10.0',
                    'dwave-networkx>=0.8.10',
                    'dwave-preprocessing>=0.3,<0.4',
                    'networkx>=2.0,<3.0',
                    'homebase>=1.0.0,<2.0.0',
                    'minorminer>=0.2.4,<0.3.0',
                    'numpy>=1.17.3,<2.0.0',
                    'dwave-tabu>=0.4.2',
                    ]

# NOTE: dwave-drivers can also be installed with `dwave install drivers`,
# and the exact requirements for `drivers` contrib package are defined in
# `dwave.system.package_info`, the `contrib` dict.
extras_require = {'drivers': ['dwave-drivers>=0.4.0,<0.5.0'],
                  }

python_requires = '>=3.6'

packages = ['dwave',
            'dwave.embedding',
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
    entry_points={
        'dwave_contrib': [
            'dwave-system = dwave.system.package_info:contrib'
        ]
    },
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=python_requires,
    zip_safe=False
)
