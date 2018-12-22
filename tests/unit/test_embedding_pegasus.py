from dwave.embedding.pegasus import find_clique_embedding
from dwave.embedding.diagnostic import is_valid_embedding
from dwave_networkx.generators.pegasus import pegasus_graph
import networkx as nx
import unittest


class TestFindClique(unittest.TestCase):
    def test_k_parameter_int(self):
        k_int = 5
        m = 3

        # Find embedding
        pg = pegasus_graph(m, coordinates=True)
        embedding = find_clique_embedding(k_int, target_graph=pg)

        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k_int), pg))

    def test_k_parameter_list(self):
        k_nodes = ['one', 'two', 'three']
        m = 4

        # Find embedding
        pg = pegasus_graph(m, coordinates=True)
        embedding = find_clique_embedding(k_nodes, target_graph=pg)

        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k_nodes), pg))

    def test_k_parameter_graph(self):
        k_graph = nx.complete_graph(10)
        m = 4

        # Find embedding
        pg = pegasus_graph(m, coordinates=True)
        embedding = find_clique_embedding(k_graph, target_graph=pg)

        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k_graph), pg))

    def test_zero_clique(self):
        k = 0
        m = 3

        # Find embedding
        pg = pegasus_graph(m)
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

    def test_valid_clique_ints(self):
        k = nx.complete_graph(55)
        m = 6

        # Find embedding
        pg = pegasus_graph(m)
        embedding = find_clique_embedding(k, target_graph=pg)

        self.assertTrue(is_valid_embedding(embedding, k, pg))

    def test_valid_clique_coord(self):
        k = nx.complete_graph(55)
        m = 6

        # Find embedding
        pg = pegasus_graph(m, coordinates=True)
        embedding = find_clique_embedding(k, target_graph=pg)

        self.assertTrue(is_valid_embedding(embedding, k, pg))

    def test_impossible_clique(self):
        k = 55
        m = 2

        # Find embedding
        pg = pegasus_graph(m)

        with self.assertRaises(ValueError):
            find_clique_embedding(k, target_graph=pg)

    def test_clique_incomplete_graph(self):
        k = 5
        m = 2

        # Nodes in a known K5 embedding
        # Note: {0: [14, 32], 1: [33, 15], 2: [16, 34], 3: [35, 17], 4: [36, 12]}
        known_embedding_nodes = {14, 32, 33, 15, 16, 34, 35, 17, 36, 12}

        # Create graph with missing nodes
        incomplete_pg = pegasus_graph(m)
        removed_nodes = set(incomplete_pg.nodes) - known_embedding_nodes
        incomplete_pg.remove_nodes_from(removed_nodes)

        # See if clique embedding is found
        embedding = find_clique_embedding(k, target_graph=incomplete_pg)
        self.assertTrue(is_valid_embedding(embedding, nx.complete_graph(k), incomplete_pg))


if __name__ == "__main__":
    unittest.main()
