# Copyright 2020 D-Wave Systems Inc.
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

from contextlib import AbstractContextManager
from numbers import Number
from typing import Tuple

import dimod
import networkx as nx

from minorminer.busclique import find_clique_embedding, busgraph_cache
from dwave.preprocessing import ScaleComposite
from dwave.system.samplers.dwave_sampler import DWaveSampler
from dwave.system.coupling_groups import coupling_groups

__all__ = ['DWaveCliqueSampler']

class _QubitCouplingComposite(dimod.ComposedSampler):
    """Composite that scales variables of a problem.

    Checks whether the per qubit (or per group) coupling range is violated for
    the QPU and rescales accordingly. Scales the variables of a binary quadratic
    model (BQM) and modifies linear and quadratic terms accordingly.

    Args:
       sampler (:obj:`~dimod.ComposedSampler`):
            A dimod sampler.
    """
    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    @dimod.decorators.nonblocking_sample_method
    def sample(self, bqm, **parameters):
        """ Scale and sample from the provided binary quadratic model.

        Problem is scaled based on the per qubit coupling range when 
        that range is exceeded. 

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`~dimod.SampleSet`
        """
        if any(('per_qubit_coupling_range' in self.child.properties.keys(),
                'per_group_coupling_range' in self.child.properties.keys())):

            if 'per_qubit_coupling_range' in self.child.properties.keys():
                min_lim = self.child.properties['per_qubit_coupling_range'][0]
                max_lim = self.child.properties['per_qubit_coupling_range'][1]
                limit_name = 'per_qubit_coupling_range'

                total_coupling_range = {v: sum(bqm.adj[v].values())
                                        for v in bqm.variables}
            else:
                min_lim = self.child.properties['per_group_coupling_range'][0]
                max_lim = self.child.properties['per_group_coupling_range'][1]
                limit_name = 'per_group_coupling_range'

                hardware_graph = self.child.to_networkx_graph()
                total_coupling_range = {_: sum(bqm.quadratic.get(edge, 0) for edge in group)
                                        for _, group in enumerate(coupling_groups(hardware_graph))}

            min_coupling_range = min(total_coupling_range.values())
            max_coupling_range = max(total_coupling_range.values())

            if (min_coupling_range < min_lim or max_coupling_range > max_lim):
                # scaling 
                inv_scalar = max(min_coupling_range / min_lim, 
                                 max_coupling_range / max_lim)
                scalar = 1.0 / inv_scalar

                bqm.scale(scalar,
                          ignored_variables=[],
                          ignored_interactions=[],
                          ignore_offset=[])

                sampleset = self.child.sample(bqm, **parameters)
                yield sampleset 

            else:
                sampleset = self.child.sample(bqm, **parameters)
                yield sampleset 
        else:
            sampleset = self.child.sample(bqm, **parameters)
            yield sampleset 

        yield sampleset 

class DWaveCliqueSampler(dimod.Sampler, AbstractContextManager):
    r"""A sampler for solving clique binary quadratic models on the D-Wave system.

    This sampler wraps
    :func:`~minorminer.busclique.find_clique_embedding` to generate embeddings
    with even chain length. These embeddings work well for dense
    binary quadratic models. For sparse models, using
    :class:`.EmbeddingComposite` with :class:`.DWaveSampler` is preferred.

    Configuration such as :term:`solver` selection is similar to that of
    :class:`.DWaveSampler`.

    Args:
        failover (bool, optional, default=False):
            Signal a failover condition if a sampling error occurs. When ``True``,
            raises :exc:`~dwave.system.exceptions.FailoverCondition` or
            :exc:`~dwave.system.exceptions.RetryCondition` on sampleset resolve
            to signal failover.

            Actual failover, i.e. selection of a new solver, has to be handled
            by the user. A convenience method :meth:`.trigger_failover` is available
            for this. Note that hardware graphs vary between QPUs, so triggering
            failover results in regenerated :attr:`~dimod.Structured.nodelist`, 
            :attr:`~dimod.Structured.edgelist`, :attr:`.properties` and 
            :attr:`.parameters`.

            .. versionchanged:: 1.16.0

               In the past, the :meth:`.sample` method was blocking and
               ``failover=True`` caused a solver failover and sampling retry.
               However, this failover implementation broke when :meth:`sample`
               became non-blocking (asynchronous), Setting ``failover=True`` had
               no effect.

        retry_interval (number, optional, default=-1):
            Ignored, but kept for backward compatibility.

            .. versionchanged:: 1.16.0

               Ignored since 1.16.0. See note for ``failover`` parameter above.

        **config:
            Keyword arguments, as accepted by :class:`.DWaveSampler`

    .. versionadded:: 1.29.0
        Support for context manager protocol.

    Note:
        The recommended way to use :class:`DWaveCliqueSampler` is from a
        `runtime context <https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers>`_:

        >>> with DWaveCliqueSampler() as sampler:   # doctest: +SKIP
        ...     sampler.sample(...)

        Alternatively, call the :meth:`~DWaveCliqueSampler.close` method to
        terminate the sampler resources:

        >>> sampler = DWaveCliqueSampler()      # doctest: +SKIP
        ...
        >>> sampler.close()     # doctest: +SKIP

    Examples:
        This example creates a BQM based on a 6-node clique (complete graph),
        with random :math:`\pm 1` values assigned to nodes, and submits it to
        a D-Wave system. Parameters for communication with the system, such
        as its URL and an authentication token, are implicitly set in a
        configuration file or as environment variables, as described in
        `Configuring Access to D-Wave Solvers <https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html>`_.

        >>> from dwave.system import DWaveCliqueSampler
        >>> import dimod
        ...
        >>> bqm = dimod.generators.ran_r(1, 6)
        ...
        >>> with DWaveCliqueSampler() as sampler:   # doctest: +SKIP
        ...     print(sampler.largest_clique_size > 5)
        ...     sampleset = sampler.sample(bqm, num_reads=100)
        True

    """
    def __init__(self, *,
                 failover: bool = False, retry_interval: Number = -1,
                 **config):
        self.child = DWaveSampler(
            failover=failover, retry_interval=retry_interval, **config)

    def close(self):
        """Close the child sampler to release system resources such as threads.

        .. note::

            The method blocks for all the currently scheduled work (sampling
            requests) to finish.

        See: :meth:`~dwave.system.samplers.dwave_sampler.DWaveSampler.close`.
        """
        self.child.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """Release system resources allocated and raise any exception triggered
        within the runtime context.
        """
        self.close()
        return None

    @property
    def parameters(self) -> dict:
        try:
            return self._parameters
        except AttributeError:
            pass

        self._parameters = parameters = self.child.parameters.copy()

        # this sampler handles scaling
        parameters.pop('auto_scale', None)
        parameters.pop('bias_range', None)
        parameters.pop('quadratic_range', None)

        return parameters

    @property
    def properties(self) -> dict:
        try:
            return self._properties
        except AttributeError:
            pass

        self._properties = dict(qpu_properties=self.child.properties)
        return self.properties

    @property
    def largest_clique_size(self) -> int:
        """The maximum number of variables that can be embedded."""
        return len(self.largest_clique())

    @property
    def qpu_linear_range(self) -> Tuple[float, float]:
        """Range of linear biases allowed by the QPU."""
        try:
            return self._qpu_linear_range
        except AttributeError:
            pass

        # get the energy range
        try:
            energy_range = tuple(self.child.properties['h_range'])
        except KeyError as err:
            # for backwards compatibility with old software solvers
            if self.child.solver.software:
                energy_range = (-2, 2)
            else:
                raise err

        self._qpu_linear_range = energy_range

        return energy_range

    @property
    def qpu_quadratic_range(self) -> Tuple[float, float]:
        """Range of quadratic biases allowed by the QPU."""
        try:
            return self._qpu_quadratic_range
        except AttributeError:
            pass

        # get the energy range
        try:
            energy_range = tuple(
                self.child.properties.get('extended_j_range',
                                          self.child.properties['j_range']))
        except KeyError as err:
            # for backwards compatibility with old software solvers
            if self.child.solver.software:
                energy_range = (-1, 1)
            else:
                raise err

        self._qpu_quadratic_range = energy_range

        return energy_range

    @property
    def target_graph(self) -> nx.Graph:
        """The QPU topology."""
        try:
            return self._target_graph
        except AttributeError:
            pass

        self._target_graph = G = self.child.to_networkx_graph()

        return G

    def clique(self, variables):
        """Return a clique embedding of the given size.

        Args:
            variables (int/collection):
                Source variables. If an integer, the variables  embedded are
                labelled `[0,n)`.

        Returns:
            dict: The clique embedding.
        """
        return find_clique_embedding(variables, self.target_graph)

    def largest_clique(self):
        """The clique embedding with the maximum number of source variables.

        Returns:
            dict: The clique embedding with the maximum number of source
            variables.

        """
        return busgraph_cache(self.target_graph).largest_clique()

    def trigger_failover(self):
        """Trigger a failover and connect to a new solver."""

        self.child.trigger_failover()

        try:
            del self._target_graph
        except AttributeError:
            pass

        try:
            del self._qpu_linear_range
        except AttributeError:
            pass

        try:
            del self._qpu_quadratic_range
        except AttributeError:
            pass

    def sample(self, bqm, chain_strength=None, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`):
                Any binary quadratic model with up to
                :attr:`.largest_clique_size` variables. This BQM is embedded
                using a clique embedding.

            chain_strength (float/mapping/callable, optional):
                Sets the coupling strength between qubits representing variables
                that form a :term:`chain`. Mappings should specify the required
                chain strength for each variable. Callables should accept the BQM
                and embedding and return a float or mapping. By default,
                ``chain_strength`` is calculated with
                :func:`~dwave.embedding.chain_strength.uniform_torque_compensation`.

            **kwargs:
                Optional keyword arguments for the sampling method, specified
                per solver in :attr:`.parameters`.
                D-Wave System Documentation's
                `solver guide <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
                describes the parameters and properties supported on the D-Wave
                system. Note that ``auto_scale`` is not supported by this
                sampler, because it scales the problem as part of the embedding
                process.

        Returns:
            :class:`~dimod.SampleSet`: Sample set constructed from a (non-blocking)
            :class:`~concurrent.futures.Future`-like object.

        """

        # some arguments should not be overwritten
        if 'auto_scale' in kwargs:
            raise TypeError("sample() got an unexpected keyword argument "
                            "'auto_scale'")
        if 'bias_range' in kwargs:
            raise TypeError("sample() got an unexpected keyword argument "
                            "'bias_range'")
        if 'quadratic_range' in kwargs:
            raise TypeError("sample() got an unexpected keyword argument "
                            "'quadratic_range'")

        # handle circular import. todo: fix
        from dwave.system.composites.embedding import FixedEmbeddingComposite

        # get the embedding
        embedding = find_clique_embedding(bqm.variables, self.target_graph,
                                          use_cache=True)

        # returns an empty embedding when the BQM is too large
        if not embedding and bqm.num_variables:
            raise ValueError("Cannot embed given BQM (size {}), sampler can "
                             "only handle problems of size {}".format(
                                len(bqm.variables), self.largest_clique_size))

        assert bqm.num_variables == len(embedding)  # sanity check

        # scaling only make sense in Ising space
        original_bqm = bqm

        if bqm.vartype is not dimod.SPIN:
            bqm = bqm.change_vartype(dimod.SPIN, inplace=False)

        sampler = FixedEmbeddingComposite(
            ScaleComposite(_QubitCouplingComposite(self.child)),
            embedding)

        if 'auto_scale' in self.child.parameters:
            kwargs['auto_scale'] = False

        sampleset = sampler.sample(bqm,
                                   bias_range=self.qpu_linear_range,
                                   quadratic_range=self.qpu_quadratic_range,
                                   chain_strength=chain_strength,
                                   **kwargs
                                   )

        # change_vartype is non-blocking
        return sampleset.change_vartype(original_bqm.vartype)
