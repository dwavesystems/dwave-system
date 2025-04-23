# Copyright 2018 D-Wave Systems Inc.
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

"""
A :ref:`dimod composite <index_dimod>` that parallelizes small problems
on a structured sampler.

The :class:`.ParallelEmbeddingComposite` class takes a problem and 
parallelizes across disjoint embeddings on a target graph.
This allows multiple independent sampling processes to be conducted in 
parallel. 

See the :ref:`index_concepts` section
for explanations of technical terms in descriptions of Ocean tools.
"""

import dimod
import dwave_networkx as dnx

import dwave.embedding
from dwave.embedding.utils import adjacency_to_edges, target_to_source
from minorminer.utils.parallel_embeddings import find_multiple_embeddings

__all__ = ["ParallelEmbeddingComposite"]


class ParallelEmbeddingComposite(dimod.Composite, dimod.Structured, dimod.Sampler):
    """Composite to parallelize sampling of a small problem on a structured sampler

    Enables parallel sampling on a structured (target) sampler by use of multiple
    disjoint embeddings.
    Embeddings, particularly for large subgraphs of large target graphs
    can be difficult to obtain. Relying on the defaults of this routine may result
    in slow embedding, see :minorminer.util.parallel_embedding for methods. Note
    that parallelization of job submissions can mitigate for network latency,
    programming time and readout time in the case of QPU samplers, subject to
    additional complexity in the embedding process.

    If a list of embeddings is not provided, the function
    :code:minorminer.utils.parallel_embeddings.find_multiple_embeddings is called
    by default to attempt a maximum number of embeddings. If the target and source
    graph match processor architecture on a Chimera, Pegasus or Zephyr then
    tiling of a known embedding in a regular pattern may be a useful embedding
    strategy and find_sublattice_embeddings can be considered.
    See minorminer.utils.parallel_embedding documentation for customizable options
    including specification of the time_out and maximum number of embeddings.
    See tests/test_parallel_embeddings.py for use cases beyond the examples
    provided.

    Args:
       sampler (:class:`~dimod.Sampler`): Structured dimod sampler such as a
           :class:`~dwave.system.samplers.DWaveSampler`.
       embeddings (list, optional): A list of embeddings. Each embedding is
           assumed to be a dictionary with source-graph nodes as keys and iterables
           on target-graph nodes to as values. The embeddings can include keys not
           required by the source graph.
       source (nx.Graph, optional): A source graph must be provided if embeddings
           are not specified. The source graph nodes should be supported by
           every embedding.
       find_embeddings (Callable, optional): A function that returns
           embeddings when it is not provided. The first two arguments are
           assumed to be the source and target graph.
       find_embeddings_args (dict, optional): key word arguments for the
           find_embeddings function. The default
           is an empty dictionary.
       one_to_iterable (bool, default=False): This parameter should be fixed to
           match the value type returned by find_embeddings. If False the
           values in every dictionary are target nodes (defining a subgraph
           embedding), these are transformed to tuples for compatibility with embed_bqm
           and unembed_sampleset. If True, the values are iterables over target
           nodes and no transformation is required.

    Examples:
       A 4-loop can be embedded on the order of N//4 times on a processor
       with N nodes.


       A NOT gate QUBO can be embedded on a Chimera[m=1,t=4] graph.
       It is possible to tile many copies of this problem on a target
       graph which is Chimera, Pegasus or Zephyr structured.
       >>> from dwave.system import DWaveSampler, EmbeddingComposite
       >>> from dwave.system import TilingComposite
       ...
       >>> sampler = EmbeddingComposite(TilingComposite(DWaveSampler(), 1, 1, 4))
       >>> Q = {(1, 1): -1, (1, 2): 2, (2, 1): 0, (2, 2): -1}
       >>> sampleset = sampler.sample_qubo(Q)
       >>> len(sampleset)> 1
       True

    See the :ref:`index_concepts` section
    for explanations of technical terms in descriptions of Ocean tools.

    """

    nodelist = None
    """list: List of active qubits for the structured solver."""

    edgelist = None
    """list: List of active couplers for the structured solver."""

    parameters = None
    """dict[str, list]: Parameters in the form of a dict."""

    properties = None
    """dict: Properties in the form of a dict."""

    children = None
    """list: The single wrapped structured sampler."""

    embeddings = []

    """list: Embeddings into each available tile on the structured solver."""

    def __init__(
        self,
        sampler,
        embeddings=None,
        source=None,
        find_embeddings=None,
        find_embeddings_args=None,
        one_to_iterable=False,
    ):

        self.parameters = sampler.parameters.copy()
        self.properties = properties = {"child_properties": sampler.properties}

        # dimod.Structured abstract base class automatically populates adjacency
        # and structure as mixins based on nodelist and edgelist
        if source is not None:
            self.nodelist = sorted(source.nodes)
            self.edgelist = sorted(source.edges)
        elif embeddings is None:
            raise ValueError("Either the source or embeddings must be provided")

        self.children = [sampler]

        if not isinstance(sampler, dimod.Structured):
            # Parallelization by embedding only makes sense for a structured
            # target.
            raise ValueError("given child sampler should be structured")

        if embeddings is not None:
            # if one_to_iterable is False:
            #    _embeddings = [{k: (v,) for k, v in emb} for emb in embeddings]
            # else:
            _embeddings = embeddings.copy()
            # Cheap consistency checks and inference of structure
            if len(embeddings) == 0:
                raise ValueError(
                    "embeddings should be a non-empty list of dictionaries"
                )

            if self.nodelist is None:
                self.nodelist = {v for v in embeddings[0].keys()}
            else:
                nodeset = set(self.nodelist)
                if not all(
                    nodeset.is_subset({v for v in emb.keys()}) for emb in embeddings
                ):
                    raise ValueError(
                        "source graph is inconsistent with the embeddings specified"
                    )
            if self.edgelist is None:
                # potentially slow, but thorough. Find the intersection graph:
                __, __, target_adjacency = sampler.structure
                edgeset = set(
                    adjacency_to_edges(
                        target_to_source(target_adjacency, embeddings[0])
                    )
                )  # Assume first.
                for emb in embeddings[1:]:
                    edgeset0 = set(
                        adjacency_to_edges(
                            target_to_source(target_adjacency, embeddings[0])
                        )
                    )
                    edgeset = edgeset.intersection(edgeset0)
                self.edgelist = sorted(edgeset)
        else:
            if source is None:
                raise ValueError("A source graph must be provided to infer embeddings")

            if find_embeddings is None:
                find_embeddings = find_multiple_embeddings
            if find_embeddings_args is None:
                find_embeddings_args = {}
            _embeddings = find_embeddings(
                source, self.child.to_networkx_graph(), **find_embeddings_args
            )
            if one_to_iterable is False:
                _embeddings = [{k: (v,) for k, v in emb.items()} for emb in _embeddings]

            print(_embeddings)
        self.children = [sampler]
        self.embeddings = properties["embeddings"] = _embeddings

    @dimod.bqm_structured
    def sample(self, bqm, **kwargs):
        """Sample from the specified binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver.

        Returns:
            :class:`~dimod.SampleSet`

        Examples:
            This example submits a simple Ising problem of just two variables on a
            D-Wave system.
            Because the problem fits in a single :term:`Chimera` unit cell, it is tiled
            across the solver's entire Chimera graph, resulting in multiple samples
            (the exact number depends on the working Chimera graph of the D-Wave system).

            >>> from dwave.system import DWaveSampler, EmbeddingComposite
            >>> from dwave.system import TilingComposite
            ...
            >>> sampler = EmbeddingComposite(TilingComposite(DWaveSampler(), 1, 1, 4))
            >>> sampleset = sampler.sample_ising({},{('a', 'b'): 1})
            >>> len(sampleset) > 1
            True

        See the :ref:`index_concepts` section
        for explanations of technical terms in descriptions of Ocean tools.

        """

        # apply the embeddings to the given problem to tile it across the child sampler
        embedded_bqm = dimod.BinaryQuadraticModel.empty(bqm.vartype)

        __, __, target_adjacency = self.child.structure
        for embedding in self.embeddings:
            embedded_bqm.update(
                dwave.embedding.embed_bqm(bqm, embedding, target_adjacency)
            )

        # solve the problem on the child system
        tiled_response = self.child.sample(embedded_bqm, **kwargs)

        responses = []

        for embedding in self.embeddings:
            embedding = {
                v: chain for v, chain in embedding.items() if v in bqm.variables
            }

            responses.append(
                dwave.embedding.unembed_sampleset(tiled_response, embedding, bqm)
            )
        if self.num_embeddings == 1:
            return responses[0]
        else:
            answer = dimod.concatenate(responses)
            answer.info.update(tiled_response.info)
            return answer

    @property
    def num_embeddings(self):
        """Number of embedding available for replicating the problem."""
        return len(self.embeddings)
