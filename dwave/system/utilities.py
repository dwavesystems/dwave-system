# Copyright 2019 D-Wave
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

from typing import Union

__all__ = [
    'anneal_schedule_with_offset',
    'classproperty',
    'common_working_graph',
    'energy_scales_custom_schedule',
    ]


def _asarray(
    argname: str,
    array_like: np.typing.ArrayLike,
    num_columns: int = 1,
    ) -> np.ndarray:
    "Coerce array-like input into a NumPy array."
    try:
        array = np.asarray_chkfinite(array_like)
    except (ValueError, TypeError) as err:
        raise ValueError(f"{argname!r}: {err}") from err

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{argname!r} must be an array-like of numbers")

    if (array.ndim > 1 and array.shape[1] != num_columns
        or array.ndim == 1 and array.ndim != num_columns):

        raise ValueError(f"{argname!r} must be a {num_columns}D array-like")

    return array

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
        anneal_schedule: Union[np.typing.ArrayLike, list, list[list[float]], None] = None,
        s: Union[np.typing.ArrayLike, list, None] = None,
        A: Union[np.typing.ArrayLike, list, None] = None,
        B: Union[np.typing.ArrayLike, list, None] = None,
        c: Union[np.typing.ArrayLike, list, None] = None
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

    if anneal_schedule is not None:

        schedule = _asarray('anneal_schedule', anneal_schedule, 4)

        s = schedule[:, 0]
        A = schedule[:, 1]
        B = schedule[:, 2]
        c = schedule[:, 3]

    else:

        s = _asarray('s', s, 1)
        A = _asarray('A', A, 1)
        B = _asarray('B', B, 1)
        c = _asarray('c', c, 1)

    A_offset = np.interp(c + anneal_offset, c, A)
    B_offset = np.interp(c + anneal_offset, c, B)
    c_offset= c + anneal_offset

    return np.column_stack((s, A_offset, B_offset, c_offset))


def energy_scales_custom_schedule(
        default_schedule: Union[np.typing.ArrayLike, list[list[float]], None] = None,
        s: Union[np.typing.ArrayLike, list[float], None] = None,
        A: Union[np.typing.ArrayLike, list[float], None] = None,
        B: Union[np.typing.ArrayLike, list[float], None] = None,
        c: Union[np.typing.ArrayLike, list[float], None] = None,
        custom_schedule: Union[np.typing.ArrayLike, list[list[float]], None] = None,
        custom_t: Union[np.typing.ArrayLike, list[float], None] = None,
        custom_s: Union[np.typing.ArrayLike, list[float], None] = None,
    ) -> np.ndarray:
    r"""Generates the energy scales for a custom anneal schedule.

    The standard annealing trajectory, published for each quantum computer on
    :ref:`this <qpu_solver_properties_specific>` page, lowers :math:`A(s)`, the
    tunneling energy, and raises :math:`B(s)`, the problem energy, according to
    a  default schedule. You can customize that schedule as described in the
    :ref:`qpu_qa_anneal_sched` section of the :ref:`qpu_annealing` page.

    This function accepts a quantum computer's default anneal schedule and your
    custom schedule (as defined by the :ref:`parameter_qpu_anneal_schedule`
    parameter), and returns the energy scales as a function of time.

    Args:
        default_schedule:
            Anneal schedule, as a 4-column |array-like|_, with column values for
            :math:`s, A, B, c` as provided by (and typically taken from) the
            spreadsheet columns of the published
            :ref:`Per-QPU Solver Properties and Schedules <qpu_solver_properties_specific>`
            page. If set, do not set parameters ``s``, ``A``, ``B``, and ``c``.
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
        custom_schedule:
            Your custom anneal schedule, as a 2-column |array-like|_, with
            column values for time, :math:`t`, and the normalized anneal
            fraction, :math:`s`. Must meet the rules described for the
            :ref:`parameter_qpu_anneal_schedule` parameter. If set, do not set
            parameters ``custom_t`` or ``custom_s``.
        custom_t: Time, :math:`t`, as a 1-dimensional |array-like|_, compliant
            with the rules described for the
            :ref:`parameter_qpu_anneal_schedule` parameter. If set
            ``anneal_schedule`` must be ``None`` and ``custom_s`` must be
            provided too.
        custom_s: Normalized anneal fraction, :math:`\frac{t}{t_a}`, as a
            1-dimensional |array-like|_, compliant with the rules described for
            the :ref:`parameter_qpu_anneal_schedule` parameter.. If set
            ``anneal_schedule`` must be ``None`` and ``custom_t`` must be
            provided too.

    Returns:
        Energy scales :math:`A(s), B(s)`, and :math:`c(s)`, as a
        :std:doc:`NumPy <numpy:index>` array with columns :math:`t, s, A, B, c`.

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
        For a default schedule provided as array :code:`schedule_qpu3`, this example
        returns the energy scales for a reverse anneal.

        >>> from dwave.system import energy_scales_custom_schedule
        ...
        >>> anneal_schedule = [[0.0, 1.0], [5, 0.45], [99, 0.45], [100, 1.0]]
        >>> energy_schedule = energy_scales_custom_schedule(
        ...     schedule_qpu3,
        ...     custom_schedule=anneal_schedule)  # doctest: +SKIP
    """
    if default_schedule is not None and (
        s is not None or A is not None or B is not None or c is not None):

            raise ValueError("Either `default_schedule` or `s, A, B, c`"
                f" can be specified. Got both inputs.")

    if default_schedule is None and (
        s is None or A is None or B is None or c is None):

            raise ValueError("If `default_schedule` is unspecified, you must"
                f" specify all of `s, A, B, c`. Not all were specified.")

    if custom_schedule is not None and (
        custom_t is not None or custom_s is not None):

            raise ValueError("Either `custom_schedule` or `custom_t, custom_s`"
                f" can be specified. Got both inputs.")

    if custom_schedule is None and (
        custom_t is None or custom_s is None):

            raise ValueError("If `custom_schedule` is unspecified, you must"
                f" specify `custom_t and custom_s`. Both were not specified.")

    if default_schedule is not None:

        schedule = _asarray('default_schedule', default_schedule, 4)

        s = schedule[:, 0]
        A = schedule[:, 1]
        B = schedule[:, 2]
        c = schedule[:, 3]

    else:

        s = _asarray('s', s, 1)
        A = _asarray('A', A, 1)
        B = _asarray('B', B, 1)
        c = _asarray('c', c, 1)

    if custom_schedule is not None:

        schedule = _asarray('custom_schedule', custom_schedule, 2)

        custom_t = schedule[:, 0]
        custom_s = schedule[:, 1]

    else:

        custom_t = _asarray('custom_t', custom_t, 1)
        custom_s = _asarray('custom_s', custom_s, 1)

    precision_s = -np.log10(np.median(np.diff(s)))
    custom_s = np.round(custom_s, decimals=int(precision_s))

    out = np.empty((0, 5))

    for index in range(1, len(custom_s)):

        if custom_s[index] == custom_s[index - 1]:  # This is a pause interval

            s_index = np.where(s == custom_s[index])
            out_interval = np.vstack((
                custom_t[index - 1],
                s[s_index],
                A[s_index],
                B[s_index],
                c[s_index])).T

        else:   # This is a sloped interval

            forward_anneal = custom_s[index] > custom_s[index - 1]

            if forward_anneal:
                interval = (s <= custom_s[index]) & (s >= custom_s[index - 1])
            else:
                interval = (s >= custom_s[index]) & (s <= custom_s[index - 1])

            t_interp = np.interp(
                s[interval],
                sorted([custom_s[index - 1], custom_s[index]]),
                [custom_t[index - 1], custom_t[index]])

            s_scales = np.stack((
                s[interval],
                A[interval],
                B[interval],
                c[interval]))

            out_interval = np.vstack((
                t_interp,
                s_scales if forward_anneal else np.flip(s_scales, axis=1))).T

            # Cut overlapped interval seams (except last interval)
            if index < len(custom_s) - 1:
                out_interval = out_interval[:-1,:]

        out = np.append(out, out_interval, axis=0)

    return out