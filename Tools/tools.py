"""
    Neural Network Visualization.
    Provides a set of API's for visualizing neural networks in 3D.

    Copyright (C) 2021  Stefan Agapie

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# Builtins
import os
import re

# Packages
from NeuralNetworkTopology import TopologyObject


def lod_3D_mesh_object_filenames(at_path: str, mesh_type: TopologyObject):
    """
    Return file paths of 3D mesh objects of specified type in
    order of greatest level of details to least level of details.

    (lod stands for level of detail).

    :param at_path: The folder containing the desired 3D mesh files.
    :param mesh_type: The desired 3D mesh object types
    :return: A list of file paths in string form.
    """

    prefix, suffix = f"{mesh_type.value}_", "T.obj"
    p_bot = re.compile(f"^{prefix}(?P<triangles>[0-9]+){suffix}$")

    triangle_counts = []
    for file in os.listdir(at_path):
        result = p_bot.match(file)
        if result:
            triangle_counts.append(int(result.group('triangles')))

    # sort by greatest level of detail first
    triangle_counts.sort(reverse=True)

    # construct file paths
    return [os.path.join(at_path, f"{prefix}{tc}{suffix}") for tc in triangle_counts]
