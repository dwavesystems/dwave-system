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
import unittest

import networkx as nx

from dwave.embedding.exceptions import MissingChainError, ChainOverlapError, DisconnectedChainError
from dwave.embedding.exceptions import InvalidNodeError, MissingEdgeError
from dwave.embedding.diagnostic import diagnose_embedding, is_valid_embedding, verify_embedding


class TestEmbeddingDiagnostic(unittest.TestCase):
    def test_missing_chain(self):
        k2 = nx.complete_graph(2)
        emb = {0: [0]}

        self.assertRaises(MissingChainError, lambda: verify_embedding(emb, k2, k2))
        self.assertFalse(is_valid_embedding(emb, k2, k2))

        etypes = set()
        for err in diagnose_embedding(emb, k2, k2):
            etypes.add(err[0])
        self.assertEqual(etypes, {MissingChainError})

    def test_chain_overlap(self):
        k2 = nx.complete_graph(2)
        emb = {0: [0], 1: [0, 1]}

        self.assertRaises(ChainOverlapError, lambda: verify_embedding(emb, k2, k2))
        self.assertFalse(is_valid_embedding(emb, k2, k2))

        etypes = set()
        for err in diagnose_embedding(emb, k2, k2):
            etypes.add(err[0])
        self.assertEqual(etypes, {ChainOverlapError})

    def test_chain_overlap_with_edges(self):
        #this is made for compatibility with minorminer; to verify that "overlapped
        #embeddings" don't report spurious MissingEdgeErrors
        k5 = nx.complete_graph(5)
        k4 = nx.complete_graph(4)
        emb = {i:[i%4, (i+1)%4] for i in k5}

        self.assertRaises(ChainOverlapError, lambda: verify_embedding(emb, k5, k4))
        self.assertFalse(is_valid_embedding(emb, k5, k4))

        etypes = set()
        for err in diagnose_embedding(emb, k5, k4):
            etypes.add(err[0])
        self.assertEqual(etypes, {ChainOverlapError})

    def test_chain_disconnect(self):
        k2 = nx.complete_graph(2)
        p3 = nx.path_graph(3)
        emb = {0: [1], 1: [0, 2]}

        self.assertRaises(DisconnectedChainError, lambda: verify_embedding(emb, k2, p3))
        self.assertFalse(is_valid_embedding(emb, k2, p3))

        etypes = set()
        for err in diagnose_embedding(emb, k2, p3):
            etypes.add(err[0])
        self.assertEqual(etypes, {DisconnectedChainError})

    def test_invalid_node(self):
        k2 = nx.complete_graph(2)
        emb = {0: [0], 1: [2]}

        self.assertRaises(InvalidNodeError, lambda: verify_embedding(emb, k2, k2))
        self.assertFalse(is_valid_embedding(emb, k2, k2))

        etypes = set()
        for err in diagnose_embedding(emb, k2, k2):
            etypes.add(err[0])
        self.assertEqual(etypes, {InvalidNodeError})

    def test_missing_edge(self):
        k2 = nx.complete_graph(2)
        e2 = nx.empty_graph(2)
        emb = {0: [0], 1: [1]}

        self.assertRaises(MissingEdgeError, lambda: verify_embedding(emb, k2, e2))
        self.assertFalse(is_valid_embedding(emb, k2, e2))

        etypes = set()
        for err in diagnose_embedding(emb, k2, e2):
            etypes.add(err[0])
        self.assertEqual(etypes, {MissingEdgeError})

    def test_valid(self):
        k2 = nx.complete_graph(2)
        emb = {0: [0], 1: [1]}

        verify_embedding(emb, k2, k2)
        self.assertTrue(is_valid_embedding(emb, k2, k2))

        for err in diagnose_embedding(emb, k2, k2):
            raise err[0](*err[1:])
