from dwave.embedding.pegasus import get_chimera_fragments, get_pegasus_coordinates, find_largest_clique_embedding
from dwave_networkx.generators.pegasus import pegasus_graph
import unittest

# Pegasus qubit offsets
VERTICAL_OFFSETS = [2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6]
HORIZONTAL_OFFSETS = [6, 6, 6, 6, 2, 2, 2, 2, 10, 10, 10, 10]


class TestGetChimeraFragments(unittest.TestCase):
    def test_empty_list(self):
        fragments = get_chimera_fragments([], VERTICAL_OFFSETS, HORIZONTAL_OFFSETS)
        self.assertEqual([], fragments)

    def test_single_horizontal_coordinate(self):
        pegasus_coord = (1, 0, 0, 0)
        fragments = get_chimera_fragments([pegasus_coord], None, HORIZONTAL_OFFSETS)

        expected_fragments = {(0, 3, 1, 0),
                              (0, 4, 1, 0),
                              (0, 5, 1, 0),
                              (0, 6, 1, 0),
                              (0, 7, 1, 0),
                              (0, 8, 1, 0)}

        self.assertEqual(expected_fragments, set(fragments))

    def test_single_vertical_coordinate(self):
        pegasus_coord = (0, 1, 3, 1)
        fragments = get_chimera_fragments([pegasus_coord], VERTICAL_OFFSETS, None)

        expected_fragments = {(7, 7, 0, 1),
                              (8, 7, 0, 1),
                              (9, 7, 0, 1),
                              (10, 7, 0, 1),
                              (11, 7, 0, 1),
                              (12, 7, 0, 1)}

        self.assertEqual(expected_fragments, set(fragments))

    def test_list_of_coordinates(self):
        pass


class TestGetPegasusCoordinates(unittest.TestCase):
    def test_empty_list(self):
        pass

    def test_single_fragment(self):
        pass

    def test_multiple_fragments_from_same_qubit(self):
        pass

    def test_mixed_fragments(self):
        pass


class TestFindLargestNativeClique(unittest.TestCase):
    def test_valid_clique(self):
        G = pegasus_graph(6)
        embedding = find_largest_clique_embedding(G)

        self.assertEqual(60, len(embedding))

        #TODO: verify connections in clique


if __name__ == "__main__":
    unittest.main()