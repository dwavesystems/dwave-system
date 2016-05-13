#!/usr/bin/env python

from distutils.core import setup

setup(name="native_embed",
      version='1.0',
      description="D-Wave Chimera Embedding Algorithm Repository",
      author="D-Wave Systems, Inc.",
      author_email="tboothby@dwavesys.com",
      packages=['chimera_embedding'],
      package_dir = {'chimera_embedding':'src'},
      scripts=['bin/nativeclique.py']
     )
