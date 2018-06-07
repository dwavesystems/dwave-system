"""who watches the watchmen?"""
import unittest

from tests.unit.mock_sampler import MockSampler

import dimod.testing as dit


class TestMockSampler(unittest.TestCase):
    def setUp(self):
        sampler = MockSampler()
        dit.assert_sampler_api(sampler)
        dit.assert_structured_api(sampler)
