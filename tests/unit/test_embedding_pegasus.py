from dwave.embedding.pegasus import find_clique_embedding
from dwave_networkx.generators.pegasus import (pegasus_graph, get_tuple_fragmentation_fn,
                                               get_tuple_defragmentation_fn)
import unittest

# Pegasus qubit offsets
VERTICAL_OFFSETS = [2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6]
HORIZONTAL_OFFSETS = [6, 6, 6, 6, 2, 2, 2, 2, 10, 10, 10, 10]


class TestTupleFragmentation(unittest.TestCase):
    def test_empty_list(self):
        # Set up fragmentation function
        G = pegasus_graph(3)
        fragment_tuple = get_tuple_fragmentation_fn(G)

        # Fragment pegasus coordinates
        fragments = fragment_tuple([])
        self.assertEqual([], fragments)

    def test_single_horizontal_coordinate(self):
        # Set up fragmentation function
        G = pegasus_graph(2)
        fragment_tuple = get_tuple_fragmentation_fn(G)

        # Fragment pegasus coordinates
        pegasus_coord = (1, 0, 0, 0)
        fragments = fragment_tuple([pegasus_coord])

        expected_fragments = {(0, 3, 1, 0),
                              (0, 4, 1, 0),
                              (0, 5, 1, 0),
                              (0, 6, 1, 0),
                              (0, 7, 1, 0),
                              (0, 8, 1, 0)}

        self.assertEqual(expected_fragments, set(fragments))

    def test_single_vertical_coordinate(self):
        # Set up fragmentation function
        G = pegasus_graph(6)
        fragment_tuple = get_tuple_fragmentation_fn(G)

        pegasus_coord = (0, 1, 3, 1)
        fragments = fragment_tuple([pegasus_coord])

        expected_fragments = {(7, 7, 0, 1),
                              (8, 7, 0, 1),
                              (9, 7, 0, 1),
                              (10, 7, 0, 1),
                              (11, 7, 0, 1),
                              (12, 7, 0, 1)}

        self.assertEqual(expected_fragments, set(fragments))

    def test_list_of_coordinates(self):
        # Set up fragmentation function
        G = pegasus_graph(6)
        fragment_tuple = get_tuple_fragmentation_fn(G)

        # Fragment pegasus coordinates
        pegasus_coords = [(1, 5, 11, 4), (0, 2, 2, 3)]
        fragments = fragment_tuple(pegasus_coords)

        expected_fragments = {(35, 29, 1, 1),
                              (35, 30, 1, 1),
                              (35, 31, 1, 1),
                              (35, 32, 1, 1),
                              (35, 33, 1, 1),
                              (35, 34, 1, 1),
                              (19, 13, 0, 0),
                              (20, 13, 0, 0),
                              (21, 13, 0, 0),
                              (22, 13, 0, 0),
                              (23, 13, 0, 0),
                              (24, 13, 0, 0)}

        self.assertEqual(expected_fragments, set(fragments))


class TestTupleDefragmentation(unittest.TestCase):
    def test_empty_list(self):
        # Set up defragmentation function
        G = pegasus_graph(2)
        defragment_tuple = get_tuple_defragmentation_fn(G)

        # De-fragment chimera coordinates
        chimera_coords = []
        pegasus_coords = defragment_tuple(chimera_coords)

        self.assertEqual([], pegasus_coords)

    def test_single_fragment(self):
        # Set up defragmentation function
        G = pegasus_graph(4)
        defragment_tuple = get_tuple_defragmentation_fn(G)

        # De-fragment chimera coordinates
        chimera_coords = [(3, 7, 0, 0)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = [(0, 1, 2, 0)]
        self.assertEqual(expected_pegasus_coords, pegasus_coords)

    def test_multiple_fragments_from_same_qubit(self):
        # Set up defragmentation function
        G = pegasus_graph(3)
        defragment_tuple = get_tuple_defragmentation_fn(G)

        # De-fragment chimera coordinates
        chimera_coords = [(9, 8, 1, 1), (9, 11, 1, 1)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = [(1, 1, 7, 1)]
        self.assertEqual(expected_pegasus_coords, pegasus_coords)

    def test_mixed_fragments(self):
        # Set up defragmenation function
        G = pegasus_graph(8)
        defragment_tuple = get_tuple_defragmentation_fn(G)

        # De-fragment chimera coordinates
        chimera_coords = [(17, 14, 0, 0), (22, 14, 0, 0), (24, 32, 1, 0), (1, 31, 0, 0)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = {(0, 2, 4, 2), (1, 4, 0, 4), (0, 5, 2, 0)}
        self.assertEqual(expected_pegasus_coords, set(pegasus_coords))

class TestFindClique(unittest.TestCase):
    #TODO: test k as a list of nodes
    def test_valid_clique(self):
        G = pegasus_graph(6)
        k = 60
        embedding = find_clique_embedding(k, target_graph=G)

        self.assertEqual(k, len(embedding))

        #TODO: verify connections in clique


if __name__ == "__main__":
    unittest.main()