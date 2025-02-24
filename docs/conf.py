# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'sphinx.ext.ifconfig'
]

autosummary_generate = True

# The suffix(es) of source filenames.
source_suffix = '.rst'

master_doc = 'index'

# General information about the project.
project = u'dwave-system'
copyright = u'2018, D-Wave Systems Inc'
author = u'D-Wave Systems Inc'

import dwave.system.package_info
version = dwave.system.package_info.__version__
release = dwave.system.package_info.__version__

language = "en"

add_module_names = False

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'sdk_index.rst']

linkcheck_retries = 2
linkcheck_anchors = False
linkcheck_ignore = [r'https://cloud.dwavesys.com/leap',  # redirects, many checks
                    r'https://www.jstor.org/stable']

pygments_style = 'sphinx'

todo_include_todos = True

modindex_common_prefix = ['dwave-system.']

doctest_global_setup = """
import dimod
from dwave.embedding import *

from unittest.mock import Mock
from dwave.system.testing import MockDWaveSampler
import dwave.system
dwave.system.DWaveSampler = Mock()
dwave.system.DWaveSampler.side_effect = MockDWaveSampler
from dwave.system import *
"""

# -- Options for HTML output ----------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads

# TODO: replace oceandocs & sysdocs_gettingstarted
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
    'networkx': ('https://networkx.github.io/documentation/stable/', None),
    'oceandocs': ('https://docs.ocean.dwavesys.com/en/stable/', None),
    'sysdocs_gettingstarted': ('https://docs.dwavesys.com/docs/latest/', None)}
