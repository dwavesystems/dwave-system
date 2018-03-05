"""who watches the watchmen?"""
import unittest

from dimod.test import SamplerAPITestCaseMixin

from tests.mock_sampler import MockSampler


class TestMockSampler(unittest.TestCase, SamplerAPITestCaseMixin):
    def setUp(self):
        self.sampler = MockSampler()
        self.sampler_factory = MockSampler
