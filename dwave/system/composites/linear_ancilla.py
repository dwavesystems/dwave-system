# coding: utf-8
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

"""Embedding composite to implement linear fields as polarized ancilla qubits.
"""

import numbers

from typing import Sequence, Mapping, Any

from collections import defaultdict

import numpy as np

from dimod.decorators import nonblocking_sample_method
import dimod


__all__ = ["LinearAncillaComposite"]


class LinearAncillaComposite(dimod.ComposedSampler, dimod.Structured):
    """Implements linear fields as polarized ancilla qubits.

    Linear field `h_i` of qubit `i` is implemented through a coupling `J_{ij}` between
    the qubit and a neighbouring qubit `j` that is fully polarized with a large flux bias.

    Args:
    child_sampler (:class:`dimod.Sampler`):
        A dimod sampler, such as a :obj:`DWaveSampler`, that has flux bias controls.

    """

    def __init__(
        self,
        child_sampler: dimod.Sampler,
    ):
        self.children = [child_sampler]
        self.parameters = child_sampler.parameters.copy()
        self.properties = dict(child_properties=child_sampler.properties.copy())
        self.nodelist = child_sampler.nodelist
        self.edgelist = child_sampler.edgelist

    def nodelist(self):
        pass  # overwritten by init

    def edgelist(self):
        pass  # overwritten by init

    children = None  # overwritten by init
    """list [child_sampler]: List containing the structured sampler."""

    parameters = None  # overwritten by init
    """dict[str, list]: Parameters in the form of a dict.

    For an instantiated composed sampler, keys are the keyword parameters
    accepted by the child sampler and parameters added by the composite.
    """

    properties = None  # overwritten by init
    """dict: Properties in the form of a dict.

    Contains the properties of the child sampler.
    """

    @nonblocking_sample_method
    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        *,
        h_tolerance: numbers.Number = 0,
        default_flux_bias_range: Sequence[float] = [-0.005, 0.005],
        **parameters,
    ):
        """Sample from the provided binary quadratic model.


        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            h_tolerance (:class:`numbers.Number`):
                Magnitude of the linear bias can be left on the qubit. Assumed to be positive. Defaults to zero.

            default_flux_bias_range (:class:`typing.Sequence`):
                Flux bias range safely accepted by the QPU, the larger the better.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`~dimod.SampleSet`

        """
        if h_tolerance < 0:
            raise ValueError("h_tolerance needs to be positive or zero")

        child = self.child
        qpu_properties = innermost_child_properties(child)
        g_target = child.to_networkx_graph()
        g_source = dimod.to_networkx_graph(bqm)
        j_range = qpu_properties["extended_j_range"]
        flux_bias_range = qpu_properties.get("flux_bias_range", default_flux_bias_range)

        # Positive couplings tend to have smaller control error,
        # we default to them if they have the same magnitude than negative couplings
        # https://docs.dwavesys.com/docs/latest/c_qpu_ice.html#overview-of-ice
        largest_j = j_range[1] if abs(j_range[1]) >= abs(j_range[0]) else j_range[0]
        largest_j_sign = np.sign(largest_j)

        # To implement the bias sign through flux bias sign,
        # we pick a range (magnitude) that we can sign-flip
        fb_magnitude = min(abs(b) for b in flux_bias_range)
        flux_biases = [0] * qpu_properties["num_qubits"]

        _bqm = bqm.copy()
        used_ancillas = defaultdict(list)
        for variable, bias in bqm.iter_linear():
            if abs(bias) <= h_tolerance:
                continue
            if abs(bias) - h_tolerance > abs(largest_j):
                return NotImplementedError(
                    "linear biases larger than the strongest coupling are not supported yet"
                )  # TODO: implement larger biases through multiple ancillas

            available_ancillas = set(g_target.adj[variable]) - set(g_source.nodes())
            if not len(available_ancillas):
                raise ValueError(f"variable {variable} has no ancillas available")
            unused_ancillas = available_ancillas - set(used_ancillas)
            if len(unused_ancillas):
                ancilla = unused_ancillas.pop()
                # bias sign is handled by the flux bias
                flux_biases[ancilla] = np.sign(bias) * largest_j_sign * fb_magnitude
                _bqm.add_interaction(
                    variable, ancilla, (abs(bias) - h_tolerance) * largest_j_sign
                )
            else:
                if qpu_properties["j_range"][0] <= bias <= qpu_properties["j_range"][1]:
                    # If j can be sign-flipped, select the least used ancilla
                    ancilla = sorted(
                        list(available_ancillas), key=lambda x: len(used_ancillas[x])
                    )[0]
                    _bqm.add_interaction(
                        variable,
                        ancilla,
                        (bias - h_tolerance * np.sign(bias))
                        * np.sign([flux_biases[ancilla]]),
                    )
                else:
                    # Ancilla sharing is limited to flux biases with a  sign
                    signed_ancillas = [
                        ancilla
                        for ancilla in available_ancillas
                        if largest_j_sign
                        == np.sign(flux_biases[ancilla]) * np.sign(bias)
                    ]
                    if not len(signed_ancillas):
                        return ValueError(
                            f"variable {variable} has no ancillas available"
                        )
                    else:
                        ancilla = sorted(
                            list(signed_ancillas), key=lambda x: len(used_ancillas[x])
                        )[0]
                        _bqm.add_interaction(
                            variable,
                            ancilla,
                            largest_j_sign * (abs(bias) - h_tolerance),
                        )

            used_ancillas[ancilla].append(variable)
            _bqm.set_linear(variable, h_tolerance * np.sign(bias))

        sampleset = self.child.sample(_bqm, flux_biases=flux_biases, **parameters)
        yield
        yield dimod.SampleSet.from_samples_bqm(
            [
                {k: v for k, v in sample.items() if k not in used_ancillas}
                for sample in sampleset.samples()
            ],
            bqm=bqm,
            info=sampleset.info.update(used_ancillas),
        )


def innermost_child_properties(sampler: dimod.Sampler) -> Mapping[str, Any]:
    """Returns the properties of the inner-most child sampler in a composite.

    Args:
        sampler: A dimod sampler

    Returns:
        properties (dict): The properties of the inner-most sampler

    """

    try:
        return innermost_child_properties(sampler.child)
    except AttributeError:
        return sampler.properties