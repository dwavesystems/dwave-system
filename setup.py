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

import os

from setuptools import setup, find_namespace_packages


# change directories so this works when called from other locations. Useful in build systems.
setup_folder_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_folder_loc)

exec(open(os.path.join(".", "dwave", "system", "package_info.py")).read())


install_requires = ['dimod>=0.12.20,<0.14.0',
                    'dwave-optimization>=0.1.0,<0.8',
                    'dwave-cloud-client>=0.12.0,<0.15.0',
                    'dwave-networkx>=0.8.10',
                    'dwave-preprocessing>=0.5.0',
                    'homebase>=1.0.0,<2.0.0',
                    'minorminer>=0.2.19,<0.3.0',    # lower bound for parallel embedding support
                    'numpy>=1.21.6',   # minimum inherited from minorminer
                    'dwave-samplers>=1.0.0',
                    'scipy>=1.7.3',
                    ]

python_requires = '>=3.9'

packages = find_namespace_packages(include=['dwave.*'])

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
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
    python_requires=python_requires,
    classifiers=classifiers,
    zip_safe=False
)
