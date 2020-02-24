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
#
# ================================================================================================

import networkx as nx

from dwave.embedding.exceptions import MissingChainError, ChainOverlapError, DisconnectedChainError
from dwave.embedding.exceptions import InvalidNodeError, MissingEdgeError


def diagnose_embedding(emb, source, target):
    """Diagnose a minor embedding.

    Produces a generator that lists all issues with the embedding. User-friendly
    variants of this function are :func:`is_valid_embedding`, which returns a
    bool, and :func:`verify_embedding`, which raises the first observed error.

    Args:
        emb (dict):
            A mapping of source nodes to arrays of target nodes as a dict
            of form {s: [t, ...], ...}, where s is a source-graph variable and t
            is a target-graph variable.

        source (list/:obj:`networkx.Graph`):
            Graph to be embedded as a NetworkX graph or a list of edges.

        target (list/:obj:`networkx.Graph`):
            Graph being embedded into as a NetworkX graph or a list of edges.

    Yields:
        Errors yielded in the form `ExceptionClass, arg1, arg2,...`, where the
        arguments following the class are used to construct the exception object,
        which are subclasses of :exc:`.EmbeddingError`.

            :exc:`.MissingChainError`, snode: a source node label that does not
            occur as a key of `emb`, or for which emb[snode] is empty.

            :exc:`.ChainOverlapError`, tnode, snode0, snode1: a target node which
            occurs in both `emb[snode0]` and `emb[snode1]`.

            :exc:`.DisconnectedChainError`, snode: a source node label whose chain
            is not a connected subgraph of `target`.

            :exc:`.InvalidNodeError`, tnode, snode: a source node label and putative
            target node label that is not a node of `target`.

            :exc:`.MissingEdgeError`, snode0, snode1: a pair of source node labels
            defining an edge that is not present between their chains.

    Examples:
        This example diagnoses an invalid embedding from a triangular source graph
        to a square target graph. A valid embedding, such as
        `emb = {0: [1], 1: [0], 2: [2, 3]}`, yields no errors.

         >>> import networkx as nx
         >>> source = nx.complete_graph(3)
         >>> target = nx.cycle_graph(4)
         >>> embedding = {0: [2], 1: [1, 'a'], 2: [2, 3]}
         >>> diagnosis = diagnose_embedding(embedding, source, target)
         >>> for problem in diagnosis:  # doctest: +SKIP
         ...     print(problem)
         (<class 'dwave.embedding.exceptions.InvalidNodeError'>, 1, 'a')
         (<class 'dwave.embedding.exceptions.ChainOverlapError'>, 2, 2, 0)

    """

    if not hasattr(source, 'edges'):
        source = nx.Graph(source)
    if not hasattr(target, 'edges'):
        target = nx.Graph(target)

    labels = {}
    embedded = set()
    overlaps = set()
    for x in source:
        try:
            embx = emb[x]
            missing_chain = len(embx) == 0
        except KeyError:
            missing_chain = True
        if missing_chain:
            yield MissingChainError, x
            continue

        all_present = True
        for q in embx:
            if q not in target:
                all_present = False
                yield InvalidNodeError, x, q
            elif x not in labels.setdefault(q, {x}):
                labels[q].add(x)
                overlaps.add(q)

        if all_present:
            embedded.add(x)
            if not nx.is_connected(target.subgraph(embx)):
                yield DisconnectedChainError, x

    for q in overlaps:
        nodes = list(labels[q])
        root = nodes[0]
        for x in nodes[1:]:
            yield ChainOverlapError, q, root, x

    yielded = nx.Graph()
    for p, q in target.subgraph(labels).edges():
        yielded.add_edges_from((x, y) for x in labels[p] for y in labels[q])

    for x, y in source.edges():
        if x == y:
            continue
        if x in embedded and y in embedded and not yielded.has_edge(x, y):
            yield MissingEdgeError, x, y


def is_valid_embedding(emb, source, target):
    """A simple (bool) diagnostic for minor embeddings.

    See :func:`diagnose_embedding` for a more detailed diagnostic and more information.

    Args:
        emb (dict): A mapping of source nodes to arrays of target nodes as a dict
            of form {s: [t, ...], ...}, where s is a source-graph variable and t
            is a target-graph variable.
        source (graph or edgelist): Graph to be embedded.
        target (graph or edgelist): Graph being embedded into.

    Returns:
        bool: True if `emb` is valid.

    """
    for _ in diagnose_embedding(emb, source, target):
        return False
    return True


def verify_embedding(emb, source, target, ignore_errors=()):
    """A simple (exception-raising) diagnostic for minor embeddings.

    See :func:`diagnose_embedding` for a more detailed diagnostic and more information.

    Args:
        emb (dict): A mapping of source nodes to arrays of target nodes as a dict
            of form {s: [t, ...], ...}, where s is a source-graph variable and t
            is a target-graph variable.
        source (graph or edgelist): Graph to be embedded
        target (graph or edgelist): Graph being embedded into

    Raises:
        EmbeddingError: A catch-all class for the following errors:

            MissingChainError: A key is missing from `emb` or the associated chain is empty.

            ChainOverlapError: Two chains contain the same target node.

            DisconnectedChainError: A chain is disconnected.

            InvalidNodeError: A chain contains a node label not found in `target`.

            MissingEdgeError: A source edge is not represented by any target edges.

    Returns:
        bool: True if no exception is raised.
    """

    for error in diagnose_embedding(emb, source, target):
        eclass = error[0]
        if eclass not in ignore_errors:
            raise eclass(*error[1:])
    return True
