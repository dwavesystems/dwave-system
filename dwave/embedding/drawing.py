# Copyright 2019 D-Wave Systems Inc.
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

from math import ceil, sqrt

from dwave_networkx import chimera_graph, draw_chimera

__all__ = ['draw_chimera_bqm']

def draw_chimera_bqm(bqm, width=None, height=None):
    """Draws a Chimera Graph representation of a Binary Quadratic Model.

    If cell width and height not provided assumes square cell dimensions.
    Throws an error if drawing onto a Chimera graph of the given dimensions fails.

    Args:
        bqm (:obj:`dimod.BinaryQuadraticModel`):
            Should be equivalent to a Chimera graph or a subgraph of a Chimera graph produced by dnx.chimera_graph.
            The nodes and edges should have integer variables as in the dnx.chimera_graph.
        width (int, optional):
            An integer representing the number of cells of the Chimera graph will be in width.
        height (int, optional):
            An integer representing the number of cells of the Chimera graph will be in height.

    Examples:
        >>> from dwave.embedding.drawing import draw_chimera_bqm
        >>> from dimod import BinaryQuadraticModel
        >>> Q={(0, 0): 2, (1, 1): 1, (2, 2): 0, (3, 3): -1, (4, 4): -2, (5, 5): -2, (6, 6): -2, (7, 7): -2,
        ... (0, 4): 2, (0, 4): -1, (1, 7): 1, (1, 5): 0, (2, 5): -2, (2, 6): -2, (3, 4): -2, (3, 7): -2}
        >>> draw_chimera_bqm(BinaryQuadraticModel.from_qubo(Q), width=1, height=1)

    """

    linear = bqm.linear.keys()
    quadratic = bqm.quadratic.keys()

    if width is None and height is None:
        # Create a graph large enough to fit the input networkx graph.
        graph_size = ceil(sqrt((max(linear) + 1) / 8.0))
        width = graph_size
        height = graph_size

    if not width or not height:
        raise Exception("Both dimensions must be defined, not just one.")

    # A background image of the same size is created to show the complete graph.
    G0 = chimera_graph(height, width, 4)
    G = chimera_graph(height, width, 4)


    # Check if input graph is chimera graph shaped, by making sure that no edges are invalid.
    # Invalid edges can also appear if the size of the chimera graph is incompatible with the input graph in cell dimensions.
    non_chimera_nodes = []
    non_chimera_edges = []
    for node in linear:
        if not node in G.nodes:
            non_chimera_nodes.append(node)
    for edge in quadratic:
        if not edge in G.edges:
            non_chimera_edges.append(edge)

    linear_set = set(linear)
    g_node_set = set(G.nodes)

    quadratic_set = set(map(frozenset, quadratic))
    g_edge_set = set(map(frozenset, G.edges))

    non_chimera_nodes = linear_set - g_node_set
    non_chimera_edges = quadratic_set - g_edge_set

    if non_chimera_nodes or non_chimera_edges:
        raise Exception("Input graph is not a chimera graph: Nodes: %s Edges: %s" % (non_chimera_nodes, non_chimera_edges))


    # Get lists of nodes and edges to remove from the complete graph to turn the complete graph into your graph.
    remove_nodes = list(g_node_set - linear_set)
    remove_edges = list(g_edge_set - quadratic_set)

    # Remove the nodes and edges from the graph.
    for edge in remove_edges:
        G.remove_edge(*edge)
    for node in remove_nodes:
        G.remove_node(node)

    node_size = 100
    # Draw the complete chimera graph as the background.
    draw_chimera(G0, node_size=node_size*0.5, node_color='black', edge_color='black')
    # Draw your graph over the complete graph to show the connectivity.
    draw_chimera(G, node_size=node_size, linear_biases=bqm.linear, quadratic_biases=bqm.quadratic,
                     width=3)
    return