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

import math

import dimod
import dwave_networkx as dnx

from minorminer.busclique import find_clique_embedding, busgraph_cache

from dwave.system.samplers.dwave_sampler import DWaveSampler

__all__ = ['DWaveCliqueSampler']


class DWaveCliqueSampler(dimod.Sampler):
    """A sampler for solving clique problems on the D-Wave system.

    The `DWaveCliqueSampler` wraps
    :func:`minorminer.busclique.find_clique_embedding` to generate embeddings
    with even chain length. These embeddings will work well for dense
    binary quadratic models. For sparse models, using
    :class:`.EmbeddingComposite` with :class:`.DWaveSampler` is preferred.

    Args:
        **config:
            Keyword arguments, as accepted by :class:`.DWaveSampler`

    """
    def __init__(self, **config):

        # get the QPU with the most qubits available, subject to other
        # constraints specified in **config
        self.child = child = DWaveSampler(order_by='-num_active_qubits',
                                          **config)

        # do some topology checking
        try:
            topology_type = child.properties['topology']['type']
            shape = child.properties['topology']['shape']
        except KeyError:
            raise ValueError("given sampler has unknown topology format")

        # We need a networkx graph with certain properties. In the
        # future it would be good for DWaveSampler to handle this.
        # See https://github.com/dwavesystems/dimod/issues/647
        if topology_type == 'chimera':
            G = dnx.chimera_graph(*shape,
                                  node_list=child.nodelist,
                                  edge_list=child.edgelist,
                                  )
        elif topology_type == 'pegasus':
            G = dnx.pegasus_graph(shape[0],
                                  node_list=child.nodelist,
                                  edge_list=child.edgelist,
                                  )
        else:
            raise ValueError("unknown topology type")

        self.target_graph = G

        # get the energy range
        try:
            self.qpu_linear_range = child.properties['h_range']
            self.qpu_quadratic_range = child.properties.get(
                'extended_j_range', child.properties['j_range'])
        except KeyError as err:
            # for backwards compatibility with old software solvers
            if child.solver.is_software:
                self.qpu_linear_range = [-2, 2]
                self.qpu_quadratic_range = [-1, 1]
            else:
                raise err

    @property
    def parameters(self):
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
    def properties(self):
        try:
            return self._properties
        except AttributeError:
            pass

        self._properties = dict(qpu_properties=self.child.properties)
        return self.properties

    @property
    def largest_clique_size(self):
        """The maximum number of variables."""
        return len(self.largest_clique())

    def largest_clique(self):
        """Return a largest-size clique embedding."""
        return busgraph_cache(self.target_graph).largest_clique()

    def sample(self, bqm, chain_strength=None, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`):
                Any binary quadratic model with up to
                :attr:`.largest_clique_size` variables. This BQM is embedded
                using a dense clique embedding.

            chain_strength (float, optional):
                The (relative) chain strength to use in the embedding. By
                default a chain strength of `1.5sqrt(N)` where `N` is the size
                of the largest clique, as returned by
                :attr:`.largest_clique_size`.

            **kwargs:
                Optional keyword arguments for the sampling method, specified
                per solver in :attr:`.DWaveCliqueSampler.parameters`.
                D-Wave System Documentation's
                `solver guide <https://docs.dwavesys.com/docs/latest/doc_solver_ref.html>`_
                describes the parameters and properties supported on the D-Wave
                system. Note that `auto_scale` is not supported by this
                sampler, because it scales the problem as part of the embedding
                process.

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

        if chain_strength is None:
            # chain length determines chain strength
            if embedding and bqm.num_interactions > 0:
                squared_j = (j ** 2 for j in bqm.quadratic.values())
                rms = math.sqrt(sum(squared_j)/bqm.num_interactions)
                chain_strength = 1.5 * rms * bqm.degrees(array=True).mean()
            else:
                chain_strength = 1  # doesn't matter

        sampler = FixedEmbeddingComposite(
            dimod.ScaleComposite(self.child),
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
