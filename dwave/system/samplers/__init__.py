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

import typing


class ResultInfoDict(typing.TypedDict, total=False):
    """Returned in ``SampleSet.info`` and ``LeapHybridNLSampler.SampleResult.info``.

    Not all fields defined below are always set, and additional might be set.
    """

    timing: dict[str, float]
    warnings: list[str]
    problem_id: str
    problem_label: str
    problem_data_id: str


from dwave.system.samplers.clique import *
from dwave.system.samplers.dwave_sampler import *
from dwave.system.samplers.leap_hybrid_sampler import *
