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


class EmbeddingError(Exception):
    """Base class for all embedding exceptions."""


class MissingChainError(EmbeddingError):
    """Raised if a node in the source graph has no associated chain.

    Args:
        snode: The source node with no associated chain.

    """
    def __init__(self, snode):
        msg = "chain for {} is empty or not contained in this embedding"
        super(MissingChainError, self).__init__(msg.format(snode))
        self.source_node = snode


class ChainOverlapError(EmbeddingError):
    """Raised if two source nodes have an overlapping chain.

    Args:
        tnode: Location where the chains overlap.
        snode0: First source node with overlapping chain.
        snode1: Second source node with overlapping chain.

    """
    def __init__(self, tnode, snode0, snode1):
        msg = "overlapped chains at target node {}: source nodes are {} and {}"
        super(ChainOverlapError, self).__init__(msg.format(tnode, snode0, snode1))
        self.target_node = tnode
        self.source_nodes = (snode0, snode1)


class DisconnectedChainError(EmbeddingError):
    """Raised if a chain is not connected in the target graph.

    Args:
        snode: The source node associated with the broken chain.

    """
    def __init__(self, snode):
        msg = "chain for {} is not connected"
        super(DisconnectedChainError, self).__init__(msg.format(snode))
        self.source_node = snode


class InvalidNodeError(EmbeddingError):
    """Raised if a chain contains a node not in the target graph.

    Args:
        snode: The source node associated with the chain.
        tnode: The node in the chain not in the target graph.

    """
    def __init__(self, snode, tnode):
        msg = "chain for {} contains a node label {} not contained in the target graph"
        super(InvalidNodeError, self).__init__(msg.format(snode, tnode))
        self.source_node = snode
        self.target_node = tnode


class MissingEdgeError(EmbeddingError):
    """Raised when two source nodes sharing an edge to not have a corresponding edge between their chains.

    Args:
        snode0: First source node.
        snode1: Second source node.

    """
    def __init__(self, snode0, snode1):
        msg = "source edge ({}, {}) is not represented by any target edge"
        super(MissingEdgeError, self).__init__(msg.format(snode0, snode1))
        self.source_nodes = (snode0, snode1)
