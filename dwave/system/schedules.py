# Copyright 2019 D-Wave Systems Inc.
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


def ramp(s, width, annealing_time):
    """Schedule with a ramp shape.

    Args:
        s (float):
            The mid-point of the ramp, as a fraction of the annealing time.

        width (float):
            The width of the ramp, as a fraction of the annealing time. Note
            that QPUs have a maximum slope.

        annealing_time (float):
            The total annealing time, in microseconds.

    Returns:
        list[2-tuple]: The points defining in a piece-wise curve in the shape of
        a ramp.

    Examples:
        This example constructs a schedule for a QPU that supports
        configuring an `h_gain_schedule`.

        >>> sampler = DWaveSampler(solver=dict(annealing_time=True,
                                               h_gain_schedule=True))
        >>> h = {v: -1 for v in sampler.nodelist}
        >>> schedule = ramp(.5, .2, sampler.properties['default_annealing_time'])
        >>> sampleset = sampler.sample_ising(h, {}, h_gain_schedule=schedule)

    """
    if s <= 0 or s >= 1:
        raise ValueError("s should be in interval (0, 1)")
    if width >= min(s, 1 - s) / 2:
        raise ValueError("given width takes curve outside of [0, 1] interval")

    return [(0, 0),
            (annealing_time * (s - width / 2), 0),
            (annealing_time * (s + width / 2), 1),
            (annealing_time, 1)]
