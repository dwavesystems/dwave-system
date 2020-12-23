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

import unittest

from dwave.system.schedules import ramp

class TestRamp(unittest.TestCase):
    def test_typical(self):
        schedule = ramp(.5, .2, 1)
        self.assertEqual(schedule, [(0, 0), (.4, 0), (.6, 1), (1, 1)])

    def test_width_exception(self):
        with self.assertRaises(ValueError):
            ramp(.1, .2, 1)  # curve would begin at (0, 0)
    
    def test_s_exception(self):
        with self.assertRaises(ValueError):
            ramp(-1, 0, 1)


