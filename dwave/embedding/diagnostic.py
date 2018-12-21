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
    """A detailed diagnostic for minor embeddings.

    This diagnostic produces a generator, which lists all issues with `emb`. The errors
    are yielded in the form

        ExceptionClass, arg1, arg2,...

    where the arguments following the class are used to construct the exception object.
    User-friendly variants of this function are :func:`is_valid_embedding`, which returns a
    bool, and :func:`verify_embedding` which raises the first observed error.  All exceptions
    are subclasses of :exc:`.EmbeddingError`.

    Args:
        emb (dict):
            Dictionary mapping source nodes to arrays of target nodes.

        source (list/:obj:`networkx.Graph`):
            Graph to be embedded as a NetworkX graph or a list of edges.

        target (list/:obj:`networkx.Graph`):
            Graph being embedded into as a NetworkX graph or a list of edges.

    Yields:
        One of:
            :exc:`.MissingChainError`, snode: a source node label that does not occur as a key of `emb`, or for which emb[snode] is empty

            :exc:`.ChainOverlapError`, tnode, snode0, snode0: a target node which occurs in both `emb[snode0]` and `emb[snode1]`

            :exc:`.DisconnectedChainError`, snode: a source node label whose chain is not a connected subgraph of `target`

            :exc:`.InvalidNodeError`, tnode, snode: a source node label and putative target node label which is not a node of `target`

            :exc:`.MissingEdgeError`, snode0, snode1: a pair of source node labels defining an edge which is not present between their chains
    """

    if not hasattr(source, 'edges'):
        source = nx.Graph(source)
    if not hasattr(target, 'edges'):
        target = nx.Graph(target)

    label = {}
    embedded = set()
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
            if label.get(q, x) != x:
                yield ChainOverlapError, q, x, label[q]
            elif q not in target:
                all_present = False
                yield InvalidNodeError, x, q
            else:
                label[q] = x
        if all_present:
            embedded.add(x)
            if not nx.is_connected(target.subgraph(embx)):
                yield DisconnectedChainError, x

    yielded = nx.Graph()
    for p, q in target.subgraph(label).edges():
        yielded.add_edge(label[p], label[q])
    for x, y in source.edges():
        if x == y:
            continue
        if x in embedded and y in embedded and not yielded.has_edge(x, y):
            yield MissingEdgeError, x, y


def is_valid_embedding(emb, source, target):
    """A simple (bool) diagnostic for minor embeddings.

    See :func:`diagnose_embedding` for a more detailed diagnostic / more information.

    Args:
        emb (dict): a dictionary mapping source nodes to arrays of target nodes
        source (graph or edgelist): the graph to be embedded
        target (graph or edgelist): the graph being embedded into

    Returns:
        bool: True if `emb` is valid.

    """
    for _ in diagnose_embedding(emb, source, target):
        return False
    return True


def verify_embedding(emb, source, target, ignore_errors=()):
    """A simple (exception-raising) diagnostic for minor embeddings.

    See :func:`diagnose_embedding` for a more detailed diagnostic / more information.

    Args:
        emb (dict): a dictionary mapping source nodes to arrays of target nodes
        source (graph or edgelist): the graph to be embedded
        target (graph or edgelist): the graph being embedded into

    Raises:
        EmbeddingError: a catch-all class for the below

        MissingChainError: in case a key is missing from `emb`, or the associated chain is empty
        ChainOverlapError: in case two chains contain the same target node
        DisconnectedChainError: in case a chain is disconnected
        InvalidNodeError: in case a chain contains a node label not found in `target`
        MissingEdgeError: in case a source edge is not represented by any target edges

    Returns:
        bool: True (if no exception is raised)
    """

    for error in diagnose_embedding(emb, source, target):
        eclass = error[0]
        if eclass not in ignore_errors:
            raise eclass(*error[1:])
    return True
