# Copyright 2024 D-Wave Inc.
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

"""Composite to implement linear coefficients with ancilla qubits biased with flux-bias offsets.
"""

import numbers

from collections import defaultdict
from typing import Sequence, Mapping, Any

import dimod
import numpy as np

from dimod.decorators import nonblocking_sample_method


__all__ = ["LinearAncillaComposite"]


class LinearAncillaComposite(dimod.ComposedSampler, dimod.Structured):
    """Implements linear biases as ancilla qubits polarized with strong flux biases.

    Linear bias :math:`h_i` of qubit :math:`i` is implemented through a coupling 
    :math:`J_{ij}` between the qubit and a neighboring qubit :math:`j` that has a 
    large flux-bias offset.

    Args:
        child_sampler (:class:`dimod.Sampler`):
            A dimod sampler, such as a :class:`~dwave.system.samplers.DWaveSampler()`,
            that has flux bias controls.

    .. versionadded:: 1.30.0
        Support for context manager protocol with :meth:`dimod.Scoped`
        implemented.

    Examples:
        This example submits a two-qubit problem consisting of linear biases with opposed signs 
        and anti-ferromagnetic coupling. A D-Wave quantum computer solves it with the fast-anneal 
        protocol using ancilla qubits to represent the linear biases.

        >>> from dwave.system import DWaveSampler, EmbeddingComposite, LinearAncillaComposite
        ...
        >>> with EmbeddingComposite(LinearAncillaComposite(DWaveSampler())) as sampler:   # doctest: +SKIP
        ...     sampleset = sampler.sample_ising({0:1, 1:-1}, {(0, 1): 1}, fast_anneal=True)
        ...     sampleset.first.energy
        -3
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
        default_flux_bias_range: tuple[float, float] = (-0.005, 0.005),
        **parameters,
    ):
        """Sample from the provided binary quadratic model.

        .. note::
            This composite does not support the :ref:`parameter_qpu_auto_scale` parameter; use the
            :class:`~dwave.preprocessing.composites.ScaleComposite` for scaling.

        Args:
            bqm (:class:`~dimod.binary.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            h_tolerance (:class:`numbers.Number`):
                Magnitude of the linear bias to be set directly on problem qubits; above this the bias
                is emulated by the flux-bias offset to an ancilla qubit. Assumed to be positive. 
                Defaults to zero.

            default_flux_bias_range (:class:`tuple`):
                Flux-bias range, as a two-tuple, supported by the QPU. The values must be large enough to
                ensure qubits remain polarized throughout the annealing process.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :class:`~dimod.SampleSet`.

        """
        if h_tolerance < 0:
            raise ValueError("h_tolerance needs to be positive or zero")

        child = self.child
        qpu_properties = _innermost_child_properties(child)
        target_graph = child.to_networkx_graph()
        source_graph = dimod.to_networkx_graph(bqm)
        extended_j_range = qpu_properties["extended_j_range"]
        # flux_bias_range is not supported at the moment
        flux_bias_range = qpu_properties.get("flux_bias_range", default_flux_bias_range)

        # Positive couplings tend to have smaller control error,
        # we default to them if they have the same magnitude than negative couplings
        # See the ICE documentation at https://docs.dwavequantum.com/en/latest/quantum_research/errors.html#ice
        largest_j = max(extended_j_range[::-1], key=abs)
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
                return ValueError(
                    "linear biases larger than the strongest coupling are not supported"
                )  # TODO: implement larger biases through multiple ancillas

            available_ancillas = set(target_graph.adj[variable]) - source_graph.nodes()
            if not len(available_ancillas):
                raise ValueError(f"variable {variable} has no ancillas available")
            unused_ancillas = available_ancillas - used_ancillas.keys()
            if len(unused_ancillas):
                ancilla = unused_ancillas.pop()
                # bias sign is handled by the flux bias
                flux_biases[ancilla] = np.sign(bias) * largest_j_sign * fb_magnitude
                _bqm.add_interaction(
                    variable, ancilla, (abs(bias) - h_tolerance) * largest_j_sign
                )
            else:
                if qpu_properties["j_range"][0] <= bias <= qpu_properties["j_range"][1]:
                    # If j can be sign-flipped, select the least used ancilla regardless of the flux bias sign
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
                    # Ancilla sharing is limited to flux biases with appropiate sign
                    signed_ancillas = [
                        ancilla
                        for ancilla in available_ancillas
                        if largest_j_sign
                        == np.sign(flux_biases[ancilla] * bias)
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


def _innermost_child_properties(sampler: dimod.Sampler) -> Mapping[str, Any]:
    """Returns the properties of the inner-most child sampler in a composite.

    Args:
        sampler: A dimod sampler

    Returns:
        properties (dict): The properties of the inner-most sampler

    """

    try:
        return _innermost_child_properties(sampler.child)
    except AttributeError:
        return sampler.properties
