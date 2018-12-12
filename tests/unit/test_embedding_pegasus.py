from dwave.embedding.pegasus import get_chimera_fragments, get_pegasus_coordinates, find_largest_native_clique
import unittest


class TestGetChimeraFragments(unittest.TestCase):
    VERTICAL_OFFSETS = [2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6]
    HORIZONTAL_OFFSETS = [6, 6, 6, 6, 2, 2, 2, 2, 10, 10, 10, 10]

    def test_empty_list(self):
        pass

    def test_single_coordinate(self):
        pass

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
        pass