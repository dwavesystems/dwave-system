from dwave.embedding.pegasus import find_clique_embedding
from dwave.embedding.diagnostic import is_valid_embedding
from dwave_networkx.generators.pegasus import (pegasus_coordinates, pegasus_graph)
import networkx as nx
import unittest


class TestFindClique(unittest.TestCase):
    def test_zero_clique(self):
        k = 0

        # Find embedding
        pg = pegasus_graph(3)
        embedding = find_clique_embedding(k, target_graph=pg)

        self.assertEqual(k, len(embedding))
        self.assertEqual({}, embedding)

    def test_one_clique(self):
        k = 1
        m = 4

        # Find embedding
        pg = pegasus_graph(m, coordinates=True)
        embedding = find_clique_embedding(k, target_graph=pg)

        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k), pg))

    def test_valid_clique_int(self):
        k = nx.complete_graph(55)
        m = 6
        pg = pegasus_graph(m)
        embedding = find_clique_embedding(k, target_graph=pg)
        converter = pegasus_coordinates(m)
        embedding_indices = {key: list(converter.ints(values)) for key, values in embedding.items()}
        self.assertTrue(is_valid_embedding(embedding_indices, nx.complete_graph(k), pg))

    def test_valid_clique_coord(self):
        k = nx.complete_graph(55)
        m = 6
        pg = pegasus_graph(m, coordinates=True)
        emb = find_clique_embedding(k, target_graph=pg)
        self.assertTrue(is_valid_embedding(emb, nx.complete_graph(k), pg))

    def test_impossible_clique(self):
        k = 55
        m = 2

        # Find embedding
        pg = pegasus_graph(m)

        with self.assertRaises(ValueError):
            find_clique_embedding(k, target_graph=pg)


if __name__ == "__main__":
    unittest.main()