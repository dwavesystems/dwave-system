import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
# equivalent to:
# from penaltymodel_cache.packaing_info import *
if _PY2:
    execfile("./dwave_micro_client_dimod/package_info.py")
else:
    exec(open("./dwave_micro_client_dimod/package_info.py").read())

install_requires = ['dwave_micro_client>=0.1',
                    'dimod>=0.3.1']
extras_require = {'test': ['mock']}

setup(
    name='dwave_micro_client_dimod',
    py_modules=['dwave_micro_client_dimod'],
    version=__version__,
    description=__description__,
    url='https://github.com/dwavesystems/dwave_micro_client_dimod',
    install_requires=install_requires,
    extras_require=extras_require
)
