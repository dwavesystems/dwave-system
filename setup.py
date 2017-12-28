from __future__ import absolute_import

import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
# equivalent to:
if _PY2:
    execfile("./dwave_micro_client_dimod/package_info.py")
else:
    exec(open("./dwave_micro_client_dimod/package_info.py").read())

install_requires = ['dimod==0.5.0',
                    'dwave_micro_client==0.1',
                    'minorminer==0.1.0',
                    'dwave_embedding_utilities']
tests_require = ['numpy', 'coverage', 'mock']
extras_require = {'all': ['numpy'],
                  'tests': tests_require,
                  'docs': ['sphinx', 'sphinx_rtd_theme', 'recommonmark']}

packages = ['dwave_micro_client_dimod']

setup(
    name='dwave_micro_client_dimod',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    url='https://github.com/dwavesystems/dwave_micro_client_dimod',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=tests_require
)
