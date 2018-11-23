import networkx as nx

class EmbeddingError(RuntimeError):
    def __init__(self, msg, *args):
        super(EmbeddingError, self).__init__(msg.format(*args))
    
class MissingChainError(EmbeddingError):
    def __init__(self, snode):
        super(MissingChainError, self).__init__("chain for {} is not contained in this embedding", snode)
        self.source_node = snode

class ChainOverlapError(EmbeddingError):
    def __init__(self, tnode, snode0, snode1):
        super(ChainOverlapError, self).__init__("overlapped chains at target node {}: source nodes are {} and {}", tnode, snode0, snode1)
        self.target_node = tnode
        self.source_nodes = (snode0, snode1)

class DisconnectedChainError(EmbeddingError):
    def __init__(self, snode):
        super(DisconnectedChainError, self).__init__("chain for {} is not connected", snode)
        self.source_node = snode

class InvalidNodeError(EmbeddingError):
    def __init__(self, snode, tnode):
        super(InvalidNodeError, self).__init__("chain for {} contains a node label {} not contained in the target graph", snode, tnode)
        self.source_node = snode
        self.target_node = tnode

class MissingEdgeError(EmbeddingError):
    def __init__(self, snode0, snode1):
        super(MissingEdgeError, self).__init__("source edge ({}, {}) is not represented by any target edge", snode0, snode1)
        self.source_nodes = (snode0, snode1)

def diagnose_embedding(emb, source, target):
    """A detailed diagnostic for minor embeddings.

    This diagnostic produces a generator, which lists all issues with `emb`. The errors
    are yielded in the form

        ExceptionClass, arg1, arg2,...

    where the arguments following the class are used to construct the exception object.
    User-friendly variants of this function are :func:`is_valid_embedding`, which returns a
    bool, and :func:`verify_embedding` which raises the first observed error.  All exceptions
    are subclasses of :class:`EmbeddingError`.

    Args:
        emb (dict): a dictionary mapping source nodes to arrays of target nodes
        source (graph or edgelist): the graph to be embedded
        target (graph or edgelist): the graph being embedded into

    Yields:
        MissingChainError, snode: a source node label that does not occur as a key of `emb`
        ChainOverlapError, tnode, snode0, snode0: a target node which occurs in both `emb[snode0]` and `emb[snode1]`
        DisconnectedChainError, snode: a source node label whose chain is not a connected subgraph of `target`
        InvalidNodeError, snode, tnode: a source node label and putative target node label which is not a node of `target`
        MissingEdgeError, snode0, snode1: a pair of source node labels defining an edge which is not present between their chains
    """

    if not hasattr(source, 'edges'):
        source = nx.Graph(source)
    if not hasattr(target, 'edges'):
        target = nx.Graph(target)

    label = {}
    for x in source:
        try:
            embx = emb[x]
        except KeyError:
            yield MissingChainError, x
        for q in embx:
            if label.get(q,x) != x:
                yield ChainOverlapError, q, x, label[q]
            elif q not in target:
                yield InvalidNodeError, q, x
            else:
                label[q] = x
        if not nx.is_connected(target.subgraph(embx)):
            yield DisconnectedChainError, x

    yielded = nx.Graph()
    for p, q in target.subgraph(label).edges():
        yielded.add_edge(label[p], label[q])
    for x, y in source.edges():
        if x == y:
            continue
        if not yielded.has_edge(x, y):
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

        MissingChainError: in case a key is missing from `emb`
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