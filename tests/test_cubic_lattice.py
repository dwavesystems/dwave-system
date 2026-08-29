# Copyright 2016 D-Wave Systems Inc.
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
from dwave.system import DWaveSampler
from networkx import grid_graph
from dwave_networkx import chimera_graph, draw_chimera_embedding
from dwave.embedding.exceptions import *
from dwave.embedding.diagnostic import diagnose_embedding, verify_embedding


class Test_cubic_lattice(unittest.TestCase):
    def test_chimera_888_v1(self):

        size = 16
        sampler = DWaveSampler()
        target = chimera_graph(size, edge_list=sampler.edgelist, node_list=sampler.nodelist)

        grid = (8, 8, 8)
        source = grid_graph(list(grid))
        bqm, embedding = cubic_lattice(target, (8, 8, 8), doping=1)

        draw_chimera_embedding(target, emb=embedding)
        plt.show()
        print(bqm)

    def test_chimera_888_v2(self):

    def test_chimera_1255(self):

    def test_pegasus_p6(self):

    def test_pegasus_p16(self):



