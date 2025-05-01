# Copyright 2019 D-Wave Systems Inc.
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

"""Utility functions."""

import os
import json
import networkx as nx
import numpy as np
import warnings

from typing import List, Union

__all__ = [
    'common_working_graph',
    'classproperty',
    'anneal_schedule_with_offset',
    ]


# taken from https://stackoverflow.com/a/39542816, licensed under CC BY-SA 3.0
# not needed in py39+
class classproperty(property):
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)


def common_working_graph(graph0, graph1):
    """Creates a graph using the common nodes and edges of two given graphs.

    This function finds the edges and nodes with common labels. Note that this
    not the same as finding the greatest common subgraph with isomorphisms.

    Args:
        graph0: (dict[dict]/:obj:`~networkx.Graph`)
            A NetworkX graph or a dictionary of dictionaries adjacency
            representation.

        graph1: (dict[dict]/:obj:`~networkx.Graph`)
            A NetworkX graph or a dictionary of dictionaries adjacency
            representation.

    Returns:
        :obj:`~networkx.Graph`: A graph with the nodes and edges common to both
        input graphs.

    Examples:

        This example creates a graph that represents a part of a particular 
        Advantage quantum computer's working graph.

        >>> import dwave_networkx as dnx
        >>> from dwave.system import DWaveSampler, common_working_graph
        ...
        >>> sampler = DWaveSampler(solver={'topology__type': 'pegasus'})
        >>> P3 = dnx.pegasus_graph(3)  
        >>> p3_working_graph = common_working_graph(P3, sampler.adjacency)   

    """
    warnings.warn("dwave.system.common_working_graph() is deprecated as of dwave-system 1.23.0 "
                  "and will be removed in dwave-system 2.0. Use networkx.intersection() instead.",
                  DeprecationWarning, stacklevel=2)

    G = nx.Graph()
    G.add_nodes_from(v for v in graph0 if v in graph1)
    G.add_edges_from((u, v) for u in graph0 for v in graph0[u]
                     if v in graph1 and u in graph1[v])

    return(G)


class FeatureFlags:
    """User environment-level Ocean feature flags pertinent to dwave-system."""

    # NOTE: This is an experimental feature. If we decide to keep it, we'll want
    # to move this machinery level up to Ocean-common.

    @staticmethod
    def get(name, default=False):
        try:
            return json.loads(os.getenv('DWAVE_FEATURE_FLAGS')).get(name, default)
        except:
            return default

    @classproperty
    def hss_solver_config_override(cls):
        return cls.get('hss_solver_config_override')


def anneal_schedule_with_offset(
        anneal_offset: float = 0.0,
        anneal_schedule: Union[np.typing.ArrayLike, List, None] = None,
        s: Union[np.typing.ArrayLike, List, None] = None,
        A: Union[np.typing.ArrayLike, List, None] = None,
        B: Union[np.typing.ArrayLike, List, None] = None,
        c: Union[np.typing.ArrayLike, List, None] = None
    ) -> np.ndarray:
    r"""Calculates the anneal schedule for a given anneal offset.

    The standard annealing trajectory, published for each quantum computer on
    :ref:`this <qpu_solver_properties_specific>` page, lowers :math:`A(s)`, the
    tunneling energy, and raises :math:`B(s)`, the problem energy, identically
    for all qubits. :ref:`Anneal offsets <qpu_qa_anneal_offsets>` enable you to
    adjust the standard annealing path per qubit.

    This function accepts a quantum computer's anneal schedule and an offset
    value, and returns the advanced or delayed schedule.

    Args:
        anneal_offset:
            Anneal-offset value for a single qubit.

        anneal_schedule:
            Anneal schedule, as a 4-column |array-like|_, with column values for
            :math:`s, A, B, c` as provided by (and typically taken from) the
            spreadsheet columns of the published
            :ref:`Per-QPU Solver Properties and Schedules <qpu_solver_properties_specific>`
            page. If set, ``anneal_offset`` is the only additional parameter
            allowed.

        s: Normalized anneal fraction, :math:`\frac{t}{t_a}`, which ranges from
            0 to 1, where :math:`t_a` is the annealing duration, as a
            1-dimensional |array-like|_. If set ``anneal_schedule`` must be
            ``None`` and values must be provided for ``A``, ``B``, and ``c``.

        A: Transverse or tunneling energy, :math:`A(s)`, as a 1-dimensional
            |array-like|_. If set ``anneal_schedule`` must be ``None`` and
            values must be provided for ``s``, ``B``, and ``c``.

        B: Energy applied to the problem Hamiltonian, :math:`B(s)`, as a
            1-dimensional |array-like|_. If set ``anneal_schedule`` must be
            ``None`` and values must be provided for ``s``, ``A``, and ``c``.

        c: Normalized annealing bias, :math:`c(s)`, as a 1-dimensional
            |array-like|_. If set ``anneal_schedule`` must be ``None`` and
            values must be provided for ``s``, ``A``, and ``B``.

    Returns:
        Offset schedules :math:`A(s), B(s)`, and :math:`c(s)`, as a
        :std:doc:`NumPy <numpy:index>` array with columns :math:`s, A, B, c`.

    Note:
        You can prepare the input schedule by downloading the schedule for your
        selected quantum computer on the
        :ref:`Per-QPU Solver Properties and Schedules <qpu_solver_properties_specific>`
        page, saving the schedule tab in CSV format, and using NumPy's
        :func:`~numpy.loadtxt` function, as here:

        .. doctest::
            :skipif: True

            >>> import numpy as np
            >>> schedule = np.loadtxt(schedule_csv_filename, delimiter=",", skiprows=1)

    Examples:

        For a schedule provided as array :code:`schedule`, this example
        returns the schedule with an offset of 0.2.

        >>> from dwave.system import anneal_schedule_with_offset
        ...
        >>> offset = 0.2
        >>> schedule_offset = anneal_schedule_with_offset(offset, schedule)  # doctest: +SKIP

    """

    if anneal_schedule is not None and (
        s is not None or A is not None or B is not None or c is not None):

            raise ValueError("Either `anneal_schedule` or `s, A, B, c`"
                f" can be specified. Got both inputs.")

    if anneal_schedule is None and (
        s is None or A is None or B is None or c is None):

            raise ValueError("If `anneal_schedule` is unspecified, you must"
                f" specify all of `s, A, B, c`. Not all were specified.")

    def _require(
            argname: str,
            array_like: np.typing.ArrayLike,
            num_columns: int = 1,
            ) -> np.ndarray:
        "Coerce the input into a NumPy array."
        try:
            array = np.asarray(array_like)
        except (ValueError, TypeError) as err:
            raise ValueError(f"`{argname}` must be an array-like") from err

        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"`{argname}` must be an array-like of numbers")

        try:
            array = np.asarray_chkfinite(array)
        except ValueError as err:
            raise ValueError(f"'{argname}' must not contain infs or NaNs") from err

        if array.ndim > 1 and array.shape[1] != num_columns or \
            array.ndim == 1 and array.ndim != num_columns:

            raise ValueError(f"'{argname}' must be a {num_columns}D array-like")

        return array

    if anneal_schedule is not None:

        schedule = _require('anneal_schedule', anneal_schedule, 4)

        s = schedule[:, 0]
        A = schedule[:, 1]
        B = schedule[:, 2]
        c = schedule[:, 3]

    else:

        s = _require('s', s, 1)
        A = _require('A', A, 1)
        B = _require('B', B, 1)
        c = _require('c', c, 1)

    A_offset = np.interp(c + anneal_offset, c, A)
    B_offset = np.interp(c + anneal_offset, c, B)
    c_offset= c + anneal_offset

    return np.column_stack((s, A_offset, B_offset, c_offset))
