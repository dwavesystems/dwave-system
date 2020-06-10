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
#

import dimod
import networkx as nx

from dwave.embedding import chimera, pegasus
from dwave.system.composites.embedding import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

__all__ = ['CliqueEmbeddingComposite']


class CliqueEmbeddingComposite(EmbeddingComposite):
    """
    """

    def __init__(self, child_sampler, scale_aware=False):

        # go looking for a DWaveSampler in the child-chain. We could handle
        # the multiple children case a la dimod.child_structure_dfs but
        # for simplicty we just assume one
        qpu = child_sampler
        while not isinstance(qpu, DWaveSampler):
            try:
                qpu = qpu.child
            except AttributeError:
                raise ValueError("child_sampler must be or wrap DWaveSampler")
        self._qpu = qpu  # save for later

        try:
            topology_type = qpu.properties['topology']['type']
            topology_shape = qpu.properties['topology']['shape']
        except KeyError:
            raise ValueError("given sampler has unknown topology format")

        # Dev note: for now we'll calculate the clique embeddings on the fly.
        # In the future we'll want some sort of caching system
        if topology_type == 'chimera':
            def find_embedding(S, T):
                k = set().union(*S)  # source nodes
                m, n, t = topology_shape
                return chimera.find_clique_embedding(k, m, n, t,
                                                     target_edges=T)

        elif topology_type == 'pegasus':
            def find_embedding(S, T):
                k = set().union(*S)

                # We need a networkx graph with certain properties. In the
                # future it would be good for DWaveSampler to handle this.
                # See https://github.com/dwavesystems/dimod/issues/647
                P = nx.Graph(T)
                P.graph['rows'] = topology_shape[0]
                P.graph['labels'] = 'int'

                return pegasus.find_clique_embedding(k, target_graph=P)

        else:
            raise ValueError("given solver has an unknown topology")

        super().__init__(child_sampler,
                         scale_aware=scale_aware,
                         find_embedding=find_embedding)
