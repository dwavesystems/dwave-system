# coding: utf-8
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

"""Embedding composites for various types of problems and application.
For example:

* :class:`.EmbeddingComposite` for a problem with arbitrary structure that likely
  requires hueristic embedding.
* :class:`.AutoEmbeddingComposite` can save unnecessary embedding for
  problems that might have a structure similar to the child sampler.
* :class:`.LazyFixedEmbeddingComposite` can benefit applications that
  resubmit a BQM with changes in some values.
"""

import itertools

from warnings import warn

import dimod
import minorminer

from dwave.embedding import (target_to_source, unembed_sampleset, embed_bqm,
                             chain_to_quadratic, EmbeddedStructure)
from dwave.system.warnings import WarningHandler, WarningAction

__all__ = ('EmbeddingComposite',
           'FixedEmbeddingComposite',
           'LazyFixedEmbeddingComposite',
           'LazyEmbeddingComposite',  # deprecated
           'AutoEmbeddingComposite',
           )


class EmbeddingComposite(dimod.ComposedSampler):
    """Maps problems to a structured sampler.

    Automatically minor-embeds a problem into a structured sampler such as a
    D-Wave quantum computer. A new minor-embedding is calculated each time one 
    of its sampling methods is called.

    Args:
        child_sampler (:class:`dimod.Sampler`):
            A dimod sampler, such as a :obj:`DWaveSampler`, that accepts
            only binary quadratic models of a particular structure.

        find_embedding (function, optional):
            A function `find_embedding(S, T, **kwargs)` where `S` and `T`
            are edgelists. The function can accept additional keyword arguments.
            Defaults to :func:`minorminer.find_embedding`.

        embedding_parameters (dict, optional):
            If provided, parameters are passed to the embedding method as
            keyword arguments.

        scale_aware (bool, optional, default=False):
            Pass chain interactions to child samplers that accept an 
            ``ignored_interactions`` parameter.

        child_structure_search (function, optional):
            A function that accepts a sampler and returns the 
            :attr:`~dimod.Structured.structure` attribute.
            Defaults to :func:`dimod.child_structure_dfs`.

    Examples:

       >>> from dwave.system import DWaveSampler, EmbeddingComposite
       ...
       >>> sampler = EmbeddingComposite(DWaveSampler())
       >>> h = {'a': -1., 'b': 2}
       >>> J = {('a', 'b'): 1.5}
       >>> sampleset = sampler.sample_ising(h, J, num_reads=100)
       >>> print(sampleset.first.energy)
       -4.5


    """
    def __init__(self, child_sampler,
                 find_embedding=minorminer.find_embedding,
                 embedding_parameters=None,
                 scale_aware=False,
                 child_structure_search=dimod.child_structure_dfs):

        self.children = [child_sampler]

        # keep any embedding parameters around until later, because we might
        # want to overwrite them
        self.embedding_parameters = embedding_parameters or {}
        self.find_embedding = find_embedding

        # set the parameters
        self.parameters = parameters = child_sampler.parameters.copy()
        parameters.update(chain_strength=[],
                          chain_break_method=[],
                          chain_break_fraction=[],
                          embedding_parameters=[],
                          return_embedding=[],
                          warnings=[],
                          )

        # set the properties
        self.properties = dict(child_properties=child_sampler.properties.copy())

        # track the child's structure. We use a dfs in case intermediate
        # composites are not structured. We could expose multiple different
        # searches but since (as of 14 june 2019) all composites have single
        # children, just doing dfs seems safe for now.
        self.target_structure = child_structure_search(child_sampler)

        self.scale_aware = bool(scale_aware)

    parameters = None  # overwritten by init
    """dict[str, list]: Parameters in the form of a dict.

    For an instantiated composed sampler, keys are the keyword parameters
    accepted by the child sampler and parameters added by the composite.
    """

    children = None  # overwritten by init
    """list [child_sampler]: List containing the structured sampler."""

    properties = None  # overwritten by init
    """dict: Properties in the form of a dict.

    Contains the properties of the child sampler.
    """

    return_embedding_default = False
    """Defines the default behavior for returning embeddings. 
    
    Sets the default for the :meth:`.sample` method's 
    ``return_embedding`` optional parameter (``kwarg``).
    """

    warnings_default = WarningAction.IGNORE
    """Defines the default behavior for warnings. 
    
    Sets the default for the :meth:`.sample` method's 
    ``warnings`` optional parameter (``kwarg``).
    """

    def sample(self, bqm, chain_strength=None,
               chain_break_method=None,
               chain_break_fraction=True,
               embedding_parameters=None,
               return_embedding=None,
               warnings=None,
               **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float/mapping/callable, optional):
                Sets the coupling strength between qubits representing variables 
                that form a :term:`chain`. Mappings should specify the required 
                chain strength for each variable. Callables should accept the BQM 
                and embedding and return a float or mapping. By default, 
                ``chain_strength`` is calculated with
                :func:`~dwave.embedding.chain_strength.uniform_torque_compensation`.

            chain_break_method (function/list, optional):
                Method or methods used to resolve chain breaks. If multiple
                methods are given, the results are concatenated and a new field
                called ``chain_break_method`` specifying the index of the method
                is appended to the sample set.
                See :func:`~dwave.embedding.unembed_sampleset` and
                :mod:`~dwave.embedding.chain_breaks`.

            chain_break_fraction (bool, optional, default=True):
                Add a ``chain_break_fraction`` field to the unembedded response 
                with the fraction of chains broken before unembedding.

            embedding_parameters (dict, optional):
                If provided, parameters are passed to the embedding method as
                keyword arguments. Overrides any embedding parameters passed
                to the constructor.

            return_embedding (bool, optional):
                If True, the embedding, chain strength, chain break method and
                embedding parameters are added to the :attr:`~dimod.SampleSet.info`
                field of the returned sample set. The default behavior is defined
                by the :attr:`return_embedding_default` attribute, which by default 
                is False.

            warnings (:class:`~dwave.system.warnings.WarningAction`, optional):
                Defines what warning action to take, if any (see the
                :ref:`warnings_system` section). The default behavior is defined
                by the :attr:`warnings_default` attribute, which by default is
                :class:`~dwave.system.warnings.IGNORE`

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`~dimod.SampleSet`

        Examples:
            See the example in :class:`.EmbeddingComposite`.

        """
        if return_embedding is None:
            return_embedding = self.return_embedding_default

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = self.target_structure

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        # get the embedding
        if embedding_parameters is None:
            embedding_parameters = self.embedding_parameters
        else:
            # we want the parameters provided to the constructor, updated with
            # the ones provided to the sample method. To avoid the extra copy
            # we do an update, avoiding the keys that would overwrite the
            # sample-level embedding parameters
            embedding_parameters.update((key, val)
                                        for key, val in self.embedding_parameters
                                        if key not in embedding_parameters)

        embedding = self.find_embedding(source_edgelist, target_edgelist,
                                        **embedding_parameters)

        if bqm and not embedding:
            raise ValueError("no embedding found")

        if not hasattr(embedding, 'embed_bqm'):
            embedding = EmbeddedStructure(target_edgelist, embedding)

        bqm_embedded = embedding.embed_bqm(bqm, chain_strength=chain_strength,
                                           smear_vartype=dimod.SPIN)

        if warnings is None:
            warnings = self.warnings_default
        elif 'warnings' in child.parameters:
            parameters.update(warnings=warnings)

        warninghandler = WarningHandler(warnings)

        warninghandler.chain_strength(bqm, embedding.chain_strength, embedding)
        warninghandler.chain_length(embedding)

        if 'initial_state' in parameters:
            # if initial_state was provided in terms of the source BQM, we want
            # to modify it to now provide the initial state for the target BQM.
            # we do this by spreading the initial state values over the
            # chains
            state = parameters['initial_state']
            parameters['initial_state'] = {u: state[v]
                                           for v, chain in embedding.items()
                                           for u in chain}

        if self.scale_aware and 'ignored_interactions' in child.parameters:

            ignored = []
            for chain in embedding.values():
                # just use 0 as a null value because we don't actually need
                # the biases, just the interactions
                ignored.extend(chain_to_quadratic(chain, target_adjacency, 0))

            parameters['ignored_interactions'] = ignored

        response = child.sample(bqm_embedded, **parameters)

        def async_unembed(response):
            # unembed the sampleset aysnchronously.

            warninghandler.chain_break(response, embedding)

            sampleset = unembed_sampleset(response, embedding, source_bqm=bqm,
                                          chain_break_method=chain_break_method,
                                          chain_break_fraction=chain_break_fraction,
                                          return_embedding=return_embedding)

            if return_embedding:
                sampleset.info['embedding_context'].update(
                    embedding_parameters=embedding_parameters,
                    chain_strength=embedding.chain_strength)

            if chain_break_fraction and len(sampleset):
                warninghandler.issue("All samples have broken chains",
                                     func=lambda: (sampleset.record.chain_break_fraction.all(), None))

            if warninghandler.action is WarningAction.SAVE:
                # we're done with the warning handler so we can just pass the list
                # off, if later we want to pass in a handler or similar we should
                # do a copy
                sampleset.info.setdefault('warnings', []).extend(warninghandler.saved)

            return sampleset

        return dimod.SampleSet.from_future(response, async_unembed)


class LazyFixedEmbeddingComposite(EmbeddingComposite, dimod.Structured):
    """Maps problems to the structure of its first given problem.

    This composite reuses the minor-embedding found for its first given problem
    without recalculating a new minor-embedding for subsequent calls of its
    sampling methods.

    Args:
        child_sampler (dimod.Sampler):
            Structured dimod sampler.

        find_embedding (function, default=:func:`minorminer.find_embedding`):
            A function ``find_embedding(S, T, **kwargs)`` where ``S`` and ``T``
            are edgelists. The function can accept additional keyword arguments.
            The function is used to find the embedding for the first problem
            solved.

        embedding_parameters (dict, optional):
            If provided, parameters are passed to the embedding method as keyword
            arguments.

    Examples:

        >>> from dwave.system import LazyFixedEmbeddingComposite, DWaveSampler
        ...
        >>> sampler = LazyFixedEmbeddingComposite(DWaveSampler())
        >>> sampler.nodelist is None  # no structure prior to first sampling
        True
        >>> __ = sampler.sample_ising({}, {('a', 'b'): -1})
        >>> sampler.nodelist  # has structure based on given problem
        ['a', 'b']

    """
        
    @property
    def nodelist(self):
        """list: Nodes available to the composed sampler."""
        try:
            return self._nodelist
        except AttributeError:
            pass

        if self.adjacency is None:
            return None

        self._nodelist = nodelist = list(self.adjacency)

        # makes it a lot easier for the user if the list can be sorted, so we
        # try
        try:
            nodelist.sort()
        except TypeError:
            # python3 cannot sort unlike types
            pass

        return nodelist

    @property
    def edgelist(self):
        """list: Edges available to the composed sampler."""
        try:
            return self._edgelist
        except AttributeError:
            pass

        adj = self.adjacency

        if adj is None:
            return None

        # remove duplicates by putting into a set
        edges = set()
        for u in adj:
            for v in adj[u]:
                try:
                    edge = (u, v) if u <= v else (v, u)
                except TypeError:
                    # Py3 does not allow sorting of unlike types
                    if (v, u) in edges:
                        continue
                    edge = (u, v)

                edges.add(edge)

        self._edgelist = edgelist = list(edges)

        # makes it a lot easier for the user if the list can be sorted, so we
        # try
        try:
            edgelist.sort()
        except TypeError:
            # python3 cannot sort unlike types
            pass

        return edgelist

    @property
    def adjacency(self):
        """dict[variable, set]: Adjacency structure for the composed sampler."""
        try:
            return self._adjacency
        except AttributeError:
            pass

        if self.embedding is None:
            return None

        self._adjacency = adj = target_to_source(self.target_structure.adjacency,
                                                 self.embedding)

        return adj

    embedding = None
    """Embedding used to map binary quadratic models to the child sampler."""

    def _fix_embedding(self, embedding):
        target_edgelist = self.target_structure.edgelist
        embedding = EmbeddedStructure(target_edgelist, embedding)

        # save the embedding and overwrite the find_embedding function
        self.embedding = embedding
        self.properties.update(embedding=embedding)

        def find_embedding(S, T):
            return embedding

        self.find_embedding = find_embedding

    def sample(self, bqm, **parameters):
        """Sample the binary quadratic model.

        On the first call of a sampling method, finds a :term:`minor-embedding`
        for the given binary quadratic model (BQM). All subsequent calls to its
        sampling methods reuse this embedding.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float/mapping/callable, optional):
                Sets the coupling strength between qubits representing variables 
                that form a :term:`chain`. Mappings should specify the required 
                chain strength for each variable. Callables should accept the BQM 
                and embedding and return a float or mapping. By default, 
                ``chain_strength`` is calculated with
                :func:`~dwave.embedding.chain_strength.uniform_torque_compensation`.

            chain_break_method (function, optional):
                Method used to resolve chain breaks during sample unembedding.
                See :func:`~dwave.embedding.unembed_sampleset`.

            chain_break_fraction (bool, optional, default=True):
                Add a ``chain_break_fraction`` field to the unembedded response with
                the fraction of chains broken before unembedding.

            embedding_parameters (dict, optional):
                If provided, parameters are passed to the embedding method as
                keyword arguments. Overrides any embedding parameters passed
                to the constructor. Only used on the first call.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`~dimod.SampleSet`

        """
        if self.embedding is None:
            # get an embedding using the current find_embedding function
            embedding_parameters = parameters.pop('embedding_parameters', None)

            if embedding_parameters is None:
                embedding_parameters = self.embedding_parameters
            else:
                # update the base parameters with the new ones provided
                embedding_parameters.update((key, val)
                                            for key, val in self.embedding_parameters
                                            if key not in embedding_parameters)

            source_edgelist = list(itertools.chain(bqm.quadratic,
                                                   ((v, v) for v in bqm.linear)))

            target_edgelist = self.target_structure.edgelist

            embedding = self.find_embedding(source_edgelist, target_edgelist,
                                            **embedding_parameters)

            self._fix_embedding(embedding)

        return super(LazyFixedEmbeddingComposite, self).sample(bqm, **parameters)


class FixedEmbeddingComposite(LazyFixedEmbeddingComposite):
    """Maps problems to a structured sampler with the specified minor-embedding.

    Args:
        child_sampler (dimod.Sampler):
            Structured dimod sampler such as a D-Wave quantum computer.

        embedding (dict[hashable, iterable], optional):
            Mapping from a source graph to the specified sampler's graph (the
            target graph).

        source_adjacency (dict[hashable, iterable]):
            Deprecated. Dictionary to describe source graph as ``{node:
            {node neighbours}}``.

        kwargs:
            See the :class:`EmbeddingComposite` class for additional keyword
            arguments. Note that ``find_embedding`` and ``embedding_parameters``
            keyword arguments are ignored.

    Examples:
        To embed a triangular problem (a problem with a three-node complete graph,
        or clique) in the Chimera topology, you need to :term:`chain` two
        qubits. This example maps triangular problems to a composed sampler
        (based on the unstructured :class:`~dimod.reference.samplers.ExactSolver`)
        with a Chimera unit-cell structure.

        >>> import dimod
        >>> import dwave_networkx as dnx
        >>> from dwave.system import FixedEmbeddingComposite
        ...
        >>> c1 = dnx.chimera_graph(1)
        >>> embedding = {'a': [0, 4], 'b': [1], 'c': [5]}
        >>> structured_sampler = dimod.StructureComposite(dimod.ExactSolver(),
        ...                                               c1.nodes, c1.edges)
        >>> sampler = FixedEmbeddingComposite(structured_sampler, embedding)
        >>> sampler.edgelist
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
    """
    def __init__(self, child_sampler, embedding=None, source_adjacency=None,
                 **kwargs):
        super(FixedEmbeddingComposite, self).__init__(child_sampler, **kwargs)

        # dev note: this entire block is to support a deprecated feature and can
        # be removed in the next major release
        if embedding is None:
            if source_adjacency is None:
                raise TypeError("either embedding or source_adjacency must be "
                                "provided")
            else:
                warn(("The source_adjacency parameter is deprecated"),
                     DeprecationWarning, stacklevel=2)

            source_edgelist = [(u, v) for u in source_adjacency for v in source_adjacency[u]]

            embedding = self.find_embedding(source_edgelist,
                                            self.target_structure.edgelist)

        self._fix_embedding(embedding)


class LazyEmbeddingComposite(LazyFixedEmbeddingComposite):
    """Deprecated. Maps problems to the structure of its first given problem.

    This class is deprecated; use the :class:`LazyFixedEmbeddingComposite` class instead.

    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.
    """
    def __init__(self, child_sampler):
        super(LazyEmbeddingComposite, self).__init__(child_sampler)
        warn("'LazyEmbeddingComposite' has been renamed to 'LazyFixedEmbeddingComposite'.", DeprecationWarning)


class AutoEmbeddingComposite(EmbeddingComposite):
    """Maps problems to a structured sampler, embedding if needed.

    This composite first tries to submit the binary quadratic model directly
    to the child sampler and only embeds if a
    :exc:`~dimod.exceptions.BinaryQuadraticModelStructureError` exception is 
    raised.

    Args:
        child_sampler (:class:`dimod.Sampler`):
            Structured dimod sampler, such as a
            :obj:`~dwave.system.samplers.DWaveSampler()`.

        find_embedding (function, optional):
            A function ``find_embedding(S, T, **kwargs)`` where ``S`` and ``T``
            are edgelists. The function can accept additional keyword arguments.
            Defaults to :func:`minorminer.find_embedding`.

        kwargs:
            See the :class:`EmbeddingComposite` class for additional keyword
            arguments.

    """
    def __init__(self, child_sampler, **kwargs):

        child_search = kwargs.get('child_structure_search',
                                  dimod.child_structure_dfs)

        def permissive_child_structure(sampler):
            try:
                return child_search(sampler)
            except ValueError:
                return None
            except (AttributeError, TypeError):  # support legacy dimod
                return None

        super(AutoEmbeddingComposite, self).__init__(child_sampler,
                                                     child_structure_search=permissive_child_structure,
                                                     **kwargs)

    def sample(self, bqm, **parameters):
        child = self.child

        # we want to pass only the parameters relevent to the child sampler
        subparameters = {key: val for key, val in parameters.items()
                         if key in child.parameters}
        try:
            return child.sample(bqm, **subparameters)
        except dimod.exceptions.BinaryQuadraticModelStructureError:
            # does not match the structure so try embedding
            pass

        return super(AutoEmbeddingComposite, self).sample(bqm, **parameters)
