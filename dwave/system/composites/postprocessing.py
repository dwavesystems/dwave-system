# Copyright 2020 D-Wave Systems Inc.
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

"""Post-processing composites."""

import dimod
import greedy

__all__ = ['SteepestDescentComposite']


class SteepestDescentComposite(dimod.ComposedSampler):
    """Runs greedy local optimization (steepest descent) on input problem,
    seeded with samples from the sampler.

    Args:
        child_sampler (:class:`dimod.Sampler`):
            A dimod sampler, such as a :class:`.DWaveSampler`.

    Examples:
       >>> from dwave.system import DWaveSampler, SteepestDescentComposite
       ...
       >>> sampler = SteepestDescentComposite(DWaveSampler())
       >>> h = {'a': -1., 'b': 2}
       >>> J = {('a', 'b'): 1.5}
       >>> sampleset = sampler.sample_ising(h, J, num_reads=100)
       >>> sampleset.first.energy
       -4.5
    """

    def __init__(self, child_sampler):
        self.children = [child_sampler]

        # set the parameters
        self.parameters = parameters = child_sampler.parameters.copy()

        # set the properties
        self.properties = dict(child_properties=child_sampler.properties.copy())

    parameters = None  # overwritten by init
    """dict[str, list]: Parameters in the form of a dict.

    For an instantiated composed sampler, keys are the keyword parameters
    accepted by the child sampler and parameters added by the composite.
    """

    children = None  # overwritten by init
    """list [child_sampler]: List containing the structured sampler."""

    properties = None  # overwritten by init
    """dict: Properties in the form of a dict.

    Contains the properties of the child sampler.
    """

    def sample(self, bqm, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:class:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :class:`dimod.SampleSet`
        """

        # solve the problem on the child system
        child = self.child

        sampleset = child.sample(bqm, **parameters)

        greedy_sampler = greedy.SteepestDescentSolver()

        return greedy_sampler.sample(bqm, initial_states=sampleset)
