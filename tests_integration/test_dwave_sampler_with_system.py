import unittest
import itertools
import random

import numpy as np
import dimod

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

try:
    DWaveSampler()
    _config_found = True
except ValueError:
    _config_found = False


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestDWaveSamplerSystem(unittest.TestCase):
    def test_typical_small(self):
        h = [0, 0, 0, 0, 0]
        J = {(0, 4): 1}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        response = DWaveSampler(profile='QPU').sample(bqm)

        self.assertFalse(np.any(response.samples_matrix == 0))
        self.assertIs(response.vartype, dimod.SPIN)

        rows, cols = response.samples_matrix.shape

        self.assertEqual(cols, 5)

    def test_with_software_exact_solver(self):

        sampler = DWaveSampler(profile='software-optimize')

        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        # plant a solution

        for v in sampler.nodelist:
            bqm.add_variable(v, .001)

        for u, v in sampler.edgelist:
            bqm.add_interaction(u, v, -1)

        resp = sampler.sample(bqm, num_reads=100)

        # the ground solution should be all spin down
        ground = dict(next(iter(resp)))

        self.assertEqual(ground, {v: -1 for v in bqm})


@unittest.skipUnless(_config_found, "no configuration found to connect to a system")
class TestEmbeddingCompositeSystem(unittest.TestCase):

    def test_bqm_bug60(self):

        bqm = dimod.BinaryQuadraticModel.from_json('{"info": {}, "linear_terms": [{"bias": 9.0, "label": "carry02"}, {"bias": 3.0, "label": "carry03"}, {"bias": 3.0, "label": "carry01"}, {"bias": 7.0, "label": "and20"}, {"bias": 7.0, "label": "and21"}, {"bias": 5.0, "label": "and22"}, {"bias": -4.0, "label": "b0"}, {"bias": 0.0, "label": "b1"}, {"bias": 0.0, "label": "b2"}, {"bias": 5.0, "label": "and02"}, {"bias": 7.0, "label": "and01"}, {"bias": 2.0, "label": "sum12"}, {"bias": 0.0, "label": "sum11"}, {"bias": 5.0, "label": "carry11"}, {"bias": 3.0, "label": "carry12"}, {"bias": 0.0, "label": "a1"}, {"bias": -4.0, "label": "a0"}, {"bias": 0.0, "label": "a2"}, {"bias": 7.0, "label": "and11"}, {"bias": 7.0, "label": "and10"}, {"bias": 7.0, "label": "and12"}], "offset": -17.5, "quadratic_terms": [{"bias": 2.0, "label_head": "and10", "label_tail": "and01"}, {"bias": 2.0, "label_head": "carry03", "label_tail": "carry12"}, {"bias": 2.0, "label_head": "and21", "label_tail": "carry11"}, {"bias": 2.0, "label_head": "a0", "label_tail": "b2"}, {"bias": -4.0, "label_head": "b2", "label_tail": "and02"}, {"bias": -4.0, "label_head": "a0", "label_tail": "and01"}, {"bias": -4.0, "label_head": "and11", "label_tail": "carry11"}, {"bias": 2.0, "label_head": "and02", "label_tail": "carry01"}, {"bias": 2.0, "label_head": "a2", "label_tail": "b0"}, {"bias": -2.0, "label_head": "and21", "label_tail": "sum12"}, {"bias": -4.0, "label_head": "carry01", "label_tail": "carry02"}, {"bias": -4.0, "label_head": "and02", "label_tail": "carry02"}, {"bias": -4.0, "label_head": "a2", "label_tail": "and22"}, {"bias": -4.0, "label_head": "a0", "label_tail": "and02"}, {"bias": 4.0, "label_head": "carry11", "label_tail": "sum11"}, {"bias": 2.0, "label_head": "sum11", "label_tail": "and02"}, {"bias": -4.0, "label_head": "b0", "label_tail": "and10"}, {"bias": -4.0, "label_head": "a1", "label_tail": "and11"}, {"bias": -4.0, "label_head": "b1", "label_tail": "and11"}, {"bias": -4.0, "label_head": "and01", "label_tail": "carry01"}, {"bias": -4.0, "label_head": "sum12", "label_tail": "carry03"}, {"bias": -2.0, "label_head": "and12", "label_tail": "sum12"}, {"bias": 2.0, "label_head": "a0", "label_tail": "b1"}, {"bias": 4.0, "label_head": "carry12", "label_tail": "sum12"}, {"bias": 2.0, "label_head": "a1", "label_tail": "b0"}, {"bias": -4.0, "label_head": "b0", "label_tail": "and20"}, {"bias": 2.0, "label_head": "and20", "label_tail": "and11"}, {"bias": 2.0, "label_head": "and12", "label_tail": "carry11"}, {"bias": -4.0, "label_head": "b2", "label_tail": "and22"}, {"bias": 2.0, "label_head": "sum11", "label_tail": "carry01"}, {"bias": -4.0, "label_head": "a2", "label_tail": "and21"}, {"bias": 2.0, "label_head": "a2", "label_tail": "b2"}, {"bias": 2.0, "label_head": "a1", "label_tail": "b1"}, {"bias": 2.0, "label_head": "sum12", "label_tail": "carry02"},{"bias": -4.0, "label_head": "b2", "label_tail": "and12"}, {"bias": -4.0, "label_head": "carry02", "label_tail": "carry03"}, {"bias": -4.0, "label_head": "b1", "label_tail": "and01"}, {"bias": 2.0, "label_head": "a0", "label_tail": "b0"}, {"bias": -4.0, "label_head": "a1", "label_tail": "and10"}, {"bias": -4.0, "label_head": "and20", "label_tail": "carry11"}, {"bias": -2.0, "label_head": "and20", "label_tail": "sum11"}, {"bias": -4.0, "label_head": "a2", "label_tail": "and20"}, {"bias": 2.0, "label_head": "and22", "label_tail": "carry12"}, {"bias": 2.0, "label_head": "and21", "label_tail": "and12"}, {"bias": -4.0, "label_head": "a1", "label_tail": "and12"}, {"bias": 2.0, "label_head": "a1", "label_tail": "b2"}, {"bias": 2.0, "label_head": "a2", "label_tail": "b1"}, {"bias": -2.0, "label_head": "and11", "label_tail": "sum11"}, {"bias": -4.0, "label_head": "carry11", "label_tail": "carry12"}, {"bias": -4.0, "label_head": "b1", "label_tail": "and21"}, {"bias": -4.0, "label_head": "and21", "label_tail": "carry12"}, {"bias": -4.0, "label_head": "sum11", "label_tail": "carry02"}, {"bias": -2.0, "label_head": "carry11", "label_tail": "sum12"}, {"bias": 2.0, "label_head": "and22", "label_tail": "carry03"}, {"bias": -4.0, "label_head": "and10", "label_tail": "carry01"}, {"bias": -4.0, "label_head": "and12", "label_tail": "carry12"}], "variable_labels": ["carry02", "carry03", "carry01", "and20", "and21", "and22", "b0", "b1", "b2", "and02", "and01", "sum12", "sum11", "carry11", "carry12", "a1", "a0", "a2", "and11", "and10", "and12"], "variable_type": "BINARY", "version": {"bqm_schema": "1.0.0", "dimod": "0.6.8"}}')

        sampler = EmbeddingComposite(DWaveSampler(profile='software-optimize'))
        resp_sw = sampler.sample(bqm, num_reads=100)

        ground_energy_sw = min(resp_sw.data_vectors['energy'])

        ground_states_sw = []
        for sample, energy in resp_sw.data(['sample', 'energy']):
            if energy > ground_energy_sw:
                break

            sample = dict(sample)

            if sample not in ground_states_sw:
                ground_states_sw.append(sample)

        resp_ex = dimod.ExactSolver().sample(bqm)

        ground_energy_ex = min(resp_ex.data_vectors['energy'])

        ground_states_ex = []
        for sample, energy in resp_ex.data(['sample', 'energy']):
            if energy > ground_energy_ex:
                break

            sample = dict(sample)

            if sample not in ground_states_ex:
                ground_states_ex.append(sample)
