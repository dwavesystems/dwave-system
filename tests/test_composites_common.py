# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest import mock

import dimod
from parameterized import parameterized_class

from dwave.system.testing import MockDWaveSampler
from dwave.system.composites import (
    CutOffComposite, PolyCutOffComposite,
    EmbeddingComposite, FixedEmbeddingComposite, LazyFixedEmbeddingComposite, AutoEmbeddingComposite,
    LinearAncillaComposite,
    TilingComposite,
    ReverseAdvanceComposite, ReverseBatchStatesComposite,
)


class MockPolySampler(dimod.PolySampler):
    parameters = None
    properties = None

    def sample_poly(self, poly, **kwargs):
        pass


@parameterized_class([
    # dwave.system.composites.cutoffcomposite
    dict(composite_cls=CutOffComposite, params=dict(cutoff=0)),
    dict(composite_cls=PolyCutOffComposite, sampler_cls=MockPolySampler, params=dict(cutoff=0)),
    # dwave.system.composites.embedding
    dict(composite_cls=EmbeddingComposite),
    dict(composite_cls=FixedEmbeddingComposite, params=dict(embedding={})),
    dict(composite_cls=LazyFixedEmbeddingComposite),
    dict(composite_cls=AutoEmbeddingComposite),
    # dwave.system.composites.linear_ancilla
    dict(composite_cls=LinearAncillaComposite),
    # dwave.system.composites.tiling
    dict(composite_cls=TilingComposite, params=dict(sub_m=1, sub_n=1)),
    # dwave.system.composites.reversecomposite
    dict(composite_cls=ReverseAdvanceComposite),
    dict(composite_cls=ReverseBatchStatesComposite),
])
class TestScoped(unittest.TestCase):
    """Test all composites defined in dwave-system are properly scoped,
    i.e. they propagate `close()` to samplers and they implement the context
    manager protocol.
    """

    def get_composite(self) -> tuple[dimod.Sampler, dimod.Composite]:
        params = getattr(self, 'params', None)
        if params is None:
            params = {}

        sampler = getattr(self, 'sampler_cls', MockDWaveSampler)()
        sampler.close = mock.MagicMock()

        composite = self.composite_cls(sampler, **params)

        return sampler, composite

    def test_close_propagation(self):
        sampler, composite = self.get_composite()

        composite.close()

        sampler.close.assert_called_once()

    def test_context_manager(self):
        sampler, composite = self.get_composite()

        with composite:
            ...

        sampler.close.assert_called_once()
