import os

import homebase

from dwave_virtual_graph import __version__

__all__ = ['cache_file']

APPNAME = 'dwave-virtual-graph'
APPAUTHOR = 'dwave-systems'
DATABASENAME = 'virtual_graph_cache_v%s.db' % __version__


def cache_file(app_name=APPNAME, app_author=APPAUTHOR, filename=DATABASENAME):
    """Returns the filename (including path) for the data cache.

    Args:
        app_name (str, optional): Default 'dwave-virtual-graph'.
        app_author (str, optional): Default 'dwave-systems'
        filename (str, optional): Default 'cache_[version].db' where
            version is the current version of dwave_virtual_graph.

    Returns:
        str: The full path to the file that can be used as a cache.

    Notes:
        Creates the directory if it does not already exist.

    """
    user_data_dir = homebase.user_data_dir(app_name=app_name, app_author=app_author, create=True)
    return os.path.join(user_data_dir, filename)
