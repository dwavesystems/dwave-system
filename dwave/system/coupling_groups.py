# Copyright 2022 D-Wave Systems Inc.
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

import dwave_networkx as dnx


def coupling_groups(hardware_graph):
    """Generate groups of couplers for which a limit on total coupling applies for each group.

    Args:
        hardware_graph (:class:`networkx.Graph`): The hardware graph of a QPU.  Note that only
            :term:`Zephyr` graphs have coupling groups.

    Yields:
        Lists of tuples, where each list is a group of couplers in ``hardware_graph``.
    """

    if hardware_graph.graph.get('family') != 'zephyr':
        return

    relabel = dnx.zephyr_coordinates(hardware_graph.graph['rows']).linear_to_zephyr

    for q in hardware_graph:
        groups = [], []
        U, W, K, J, Z = relabel(q)

        for p in hardware_graph[q]:
            u, w, k, j, z = relabel(p)
            if U != u:
                groups[2*Z+J+1-w].append((p, q))
            elif J != j:
                groups[Z+J-z].append((p, q))
            else:
                groups[(z-Z)//2].append((p, q))

        yield from groups
