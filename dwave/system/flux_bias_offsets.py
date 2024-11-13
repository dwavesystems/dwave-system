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

import warnings


def get_flux_biases(*args, **kwargs):
    """Removed. Used to get the flux bias offsets for sampler and embedding,
    but now just returns an empty dictionary.

    .. versionremoved:: 1.28.0
        Due to improved calibration of newer QPUs, balancing chains with
        the flux biases, as it used to be implemented by this function (and the
        obsoleted package ``dwave-drivers``) is no longer supported.

        To calibrate chains for residual biases, follow the instructions in the
        `shimming tutorial <https://github.com/dwavesystems/shimming-tutorial>`_.

        This function (and its submodule) will be completely removed in
        dwave-system 2.0.

    Returns:
        dict:
            An empty dict, since dwave-system 1.28.
            Flux biases are not calculated/set for nodes/chains anymore.
    """

    warnings.warn(
        "'get_flux_biases' functionality is removed due to improved calibration "
        "of newer QPUs and in future will raise an exception; if needed, "
        "follow the instructions in the shimming tutorial at "
        "https://github.com/dwavesystems/shimming-tutorial instead. ",
        DeprecationWarning, stacklevel=2
    )

    return {}
