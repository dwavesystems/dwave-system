# Copyright 2025 D-Wave
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

from typing import Optional

import networkx as nx

import dimod
import dwave.embedding

from dwave.embedding.utils import adjacency_to_edges, target_to_source
from minorminer.utils.parallel_embeddings import find_multiple_embeddings

__all__ = ["ParallelEmbeddingComposite"]


class ParallelEmbeddingComposite(dimod.Composite, dimod.Structured, dimod.Sampler):
    """Composite to parallelize sampling of a small problem on a structured sampler

    Enables parallel sampling on a (target) sampler by use of multiple disjoint
    embeddings. If a list of embeddings is not provided, the function
    :func:`~minorminer.utils.parallel_embeddings.find_multiple_embeddings` is called
    by default to attempt a maximum number of embeddings. If the target and source
    graph match processor architecture on a Chimera, Pegasus or Zephyr then
    tiling of a known embedding in a regular pattern may be a useful embedding
    strategy and find_sublattice_embeddings can be considered.
    See :mod:`~minorminer.utils.parallel_embeddings` documentation for customizable options
    including specification of the time_out and maximum number of embeddings.
    See ``tests/test_parallel_embeddings.py`` for use cases beyond the examples
    provided.

    Embeddings, particularly for large subgraphs of large target graphs
    can be difficult to obtain. Relying on the defaults of this routine may result
    in slow embedding, see :mod:`~minorminer.util.parallel_embeddings` for methods. Note
    that parallelization of job submissions can mitigate for network latency,
    programming time and readout time in the case of QPU samplers, subject to
    additional complexity in the embedding process.

    Args:
       child_sampler (:class:`~dimod.Sampler`): dimod sampler such as a
           :class:`~dwave.system.samplers.DWaveSampler`.

       embeddings (list, optional): A list of embeddings. Each embedding is
           assumed to be a dictionary with source-graph nodes as keys and iterables
           on target-graph nodes to as values. The embeddings can include keys not
           required by the source graph. Note that one_to_iterable is ignored
           (assumed True).

       source (nx.Graph, optional): A source graph must be provided if embeddings
           are not specified. The source graph nodes should be supported by
           every embedding.

       embedder (Callable, optional): A function that returns
           embeddings when it is not provided. The first two arguments are
           assumed to be the source and target graph.

       embedder_kwargs (dict, optional): keyword arguments for the
           embedder function. The default is an empty dictionary.

       one_to_iterable (bool, default=False): This parameter should be fixed to
           match the value type returned by embedder. If False the
           values in every dictionary are target nodes (defining a subgraph
           embedding), these are transformed to tuples for compatibility with embed_bqm
           and unembed_sampleset. If True, the values are iterables over target
           nodes and no transformation is required.

       child_structure_search (function, optional):
           A function that accepts a sampler and returns the
           :attr:`~dimod.Structured.structure` attribute.
           Defaults to :func:`dimod.child_structure_dfs`.

    Raises:
        ValueError: If the `child_sampler` is not structured, and the structure
           cannot be inferred from `child_structure_search`.
           If neither embeddings, nor a source graph, are provided.
           If the embeddings provided are an empty list, or no embeddings are found.
           If embeddings and source graph nodes are inconsistent.
           If embeddings and target graph nodes are inconsistent.

    Examples:

        This example submits a simple Ising problem of just two variables on a
        D-Wave system. We use the default subgraph embedder finding a maximum
        number of embeddings. Note that searching for O(1000) of embeddings takes
        several seconds.

        >>> from dwave.system import DWaveSampler
        >>> from dwave.system import ParallelEmbeddingComposite
        >>> from networkx import from_edgelist
        >>> embedder_kwargs = {'max_num_emb': None}  # Without this, only 1 embedding will be sought
        >>> source = from_edgelist([('a', 'b')])
        >>> qpu = DWaveSampler()
        >>> sampler = ParallelEmbeddingComposite(qpu, source=source, embedder_kwargs=embedder_kwargs)
        >>> sampleset = sampler.sample_ising({},{('a', 'b'): 1}, num_reads=1)
        >>> len(sampleset) > 1  # Equal to the number of parallel embeddings
        True

        If an embedding can be found for a Chimera tile, we can try many
        dispacements on a target QPU graph (tiling). If all variables on the Chimera tile
        are used, and the target graph is defect free, this allows an optimal
        parallelization. Note that find_sublattice_embeddings should only be preferred
        to the default find_multiple_embeddings where the source and target graph have a
        special lattice relationship. Finding a large set of disjoint chimera cells within
        a typical processor graph can take several seconds.
        See tests/ for other examples.

        >>> from dwave.system import DWaveSampler
        >>> from dwave.system import ParallelEmbeddingComposite
        >>> from dwave_networkx import chimera_graph
        >>> from minorminer.utils.parallel_embeddings import find_sublattice_embeddings
        >>> source = tile = chimera_graph(1, 1, 4)  # A 1:1 mapping assumed
        >>> qpu = DWaveSampler()
        >>> embedder = find_sublattice_embeddings
        >>> embedder_kwargs = {'max_num_emb': None, 'tile': tile}
        >>> sampler = ParallelEmbeddingComposite(qpu, source=source, embedder=embedder, embedder_kwargs=embedder_kwargs)
        >>> J = {e: -1 for e in tile.edges}  # A ferromagnet on the Chimera tile.
        >>> sampleset = sampler.sample_ising({}, J, num_reads=1)
        >>> len(sampleset) > 1  # Equal to the number of parallel embeddings
        True

        Consider use of :func:`~dwave_networkx.drawing.draw_parallel_embeddings` for visualization of the
        embeddings found (``embeddings=sampler.embeddings`` over ``target=qpu.to_networkx_graph()``).

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
        child_sampler,
        *,
        embeddings=None,
        source=None,
        embedder=None,
        embedder_kwargs=None,
        one_to_iterable=False,
        child_structure_search=dimod.child_structure_dfs,
    ):
        self.parameters = child_sampler.parameters.copy()
        self.properties = properties = {"child_properties": child_sampler.properties}
        self.target_structure = child_structure_search(child_sampler)

        # dimod.Structured abstract base class automatically populates adjacency
        # and structure as mixins based on nodelist and edgelist
        if source is not None:
            self.nodelist = list(source.nodes)
            self.edgelist = list(source.edges)
        elif embeddings is None:
            raise ValueError("Either the source or embeddings must be provided")

        self.children = [child_sampler]
        target_nodelist, __, target_adjacency = self.target_structure
        if embeddings is not None:
            _embeddings = embeddings.copy()
            # Computationally cheap consistency checks, and inference of structure
            if len(_embeddings) == 0:
                raise ValueError(
                    "embeddings should be a non-empty list of dictionaries"
                )

            # Target graph consistency:
            nodelist = [v for emb in _embeddings for c in emb.values() for v in c]
            nodeset = set(nodelist)
            if len(nodelist) != len(nodeset):
                raise ValueError(
                    "embedding contains a non-disjoint embedding (target nodes reused)"
                )
            if not nodeset.issubset(target_nodelist):
                raise ValueError("embedding contains invalid target nodes")

            # Source graph consistency
            if self.nodelist is None:
                self.nodelist = list(embeddings[0].keys())
            else:
                nodeset = set(self.nodelist)
                if not all(nodeset.issubset(emb) for emb in embeddings):
                    raise ValueError(
                        "source graph is inconsistent with the embeddings specified"
                    )
            if self.edgelist is None:
                # Find the intersection graph (slow but thorough):
                edgeset = set(
                    adjacency_to_edges(
                        target_to_source(target_adjacency, embeddings[0])
                    )
                )
                for emb in embeddings[1:]:
                    edgeset0 = set(
                        adjacency_to_edges(target_to_source(target_adjacency, emb))
                    )
                    edgeset = edgeset.intersection(edgeset0)
                self.edgelist = list(edgeset)
            # could check viability of edgelist (valid embeddings), but this is slow and not the job of the composite.
        else:
            if source is None:
                raise ValueError("A source graph must be provided to infer embeddings")

            if embedder is None:
                embedder = find_multiple_embeddings
            if embedder_kwargs is None:
                embedder_kwargs = {}

            # The child_sampler may not preserve the graphical structure required
            # by the embedder. These might be passed as supplementary arguments.
            if hasattr(child_sampler, "to_networkx_graph"):
                _embeddings = embedder(
                    source, child_sampler.to_networkx_graph(), **embedder_kwargs
                )
            else:
                _embeddings = embedder(
                    source, nx.Graph(target_adjacency), **embedder_kwargs
                )

            if not one_to_iterable:
                _embeddings = [{k: (v,) for k, v in emb.items()} for emb in _embeddings]

            if len(_embeddings) == 0:
                raise ValueError(
                    "No embeddings found: consider changing the embedder or its parameters."
                )

        self.embeddings = properties["embeddings"] = _embeddings

    @dimod.bqm_structured
    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        chain_strength: Optional[float] = None,
        **kwargs,
    ) -> dimod.SampleSet:
        """Sample from the specified binary quadratic model. Samplesets are
        concatenated together in the the same order as the embeddings class variable,
        the info field is returned from the child sampler unmodified.

        If the bqm or chain_strength varies by solver, or if a list of samplests
        is desired as the output, use :code:`sample_multiple`.

        Args:
            bqm:
                Binary quadratic model to be sampled from.

            chain_strength:
                The chain strength parameter of the bqm.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver.

        Returns:
            :class:`~dimod.SampleSet`

        Examples:
            See class examples.
        """
        chain_strengths = [chain_strength] * self.num_embeddings
        bqms = [bqm] * self.num_embeddings
        if "initial_state" in kwargs:
            kwargs["initial_states"] = [
                kwargs.pop("initial_state")
            ] * self.num_embeddings
        responses, info = self.sample_multiple(bqms, chain_strengths, **kwargs)

        if self.num_embeddings == 1:
            return responses[0]

        answer = dimod.concatenate(responses)
        answer.info.update(info)
        return answer

    def sample_multiple(
        self,
        bqms: list[dimod.BinaryQuadraticModel],
        chain_strengths: Optional[list] = None,
        initial_states: Optional[list] = None,
        **kwargs,
    ) -> tuple[list[dimod.SampleSet], dict]:
        """Sample from the specified binary quadratic models.

        Samplesets are returned for every embedding, the binary quadratic model
        solved on each embedding needn't be identical. Keyword arguments are passed
        unmodified to the child sampler, with the exception of
        `initial_states` (one state per embedding) which is composed to a
        single `initial_state` parameter for the child sampler analogous to
        the bqm composition.

        Args:
            bqms:
                Binary quadratic models to be sampled from. A list that
                should be ordered to match ``self.embeddings``.

            chain_strengths:
                The chain strength parameters for each bqm. A list that
                should be ordered to match ``self.embeddings``.

            initial_states:
                initial state for each bqm. A list that should be ordered
                to match ``self.embeddings``.

            **kwargs:
                Optional keyword arguments for the sampling method, specified per solver.

        Returns:
            A typle consisting of:
            1. A list of :class:`~dimod.SampleSet`, one per embedding
            2. The info field returned by the child sampler
        Examples:
            See class examples.
        """

        # apply the embeddings to the given problem to tile it across the child sampler
        embedded_bqm = dimod.BinaryQuadraticModel.empty(bqms[0].vartype)

        __, __, target_adjacency = self.target_structure
        if not chain_strengths:
            chain_strengths = [None] * self.num_embeddings

        if initial_states is not None and any(i_s for i_s in initial_states):
            kwargs["initial_state"] = {
                u: state[v]
                for embedding, state in zip(self.embeddings, initial_states)
                for v, chain in embedding.items()
                for u in chain
            }

        for embedding, bqm, chain_strength in zip(
            self.embeddings, bqms, chain_strengths
        ):
            embedded_bqm.update(
                dwave.embedding.embed_bqm(
                    bqm, embedding, target_adjacency, chain_strength=chain_strength
                )
            )

        # solve the problem on the child system
        tiled_response = self.child.sample(embedded_bqm, **kwargs)

        responses = []
        for embedding, bqm in zip(self.embeddings, bqms):
            responses.append(
                dwave.embedding.unembed_sampleset(tiled_response, embedding, bqm)
            )

        return responses, tiled_response.info

    @property
    def num_embeddings(self):
        """Number of embedding available for replicating the problem."""
        return len(self.embeddings)
