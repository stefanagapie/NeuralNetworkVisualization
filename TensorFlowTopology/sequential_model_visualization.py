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
from typing import List, Tuple

# Third party - Panda3D
from panda3d.core import NodePath, LVecBase3f
from direct.showbase.ShowBase import ShowBase

# Third party - TensorFlow
import tensorflow as tf
from tensorflow.keras import Sequential

# Third party - Numpy
import numpy as np

# Packages
from NeuralNetworkTopology import TopologyObject
from NeuralNetworkTopology import TopologyAlignment
from NeuralNetworkTopology import SymmetricTopologyDelegate
from Tools import lod_3D_mesh_object_filenames


class TFSequentialModelSymmetricTopologyDelegate(SymmetricTopologyDelegate):

    def __init__(self,
                 model: Sequential, base: ShowBase,
                 neuron_path: str, edge_path: str,
                 neuron_spacing: int = 6,
                 layer_spacing: int = 45,
                 alignment: TopologyAlignment = TopologyAlignment.CENTER,
                 neuron_dimensions: LVecBase3f = LVecBase3f(1, 1, 1)):
        """
        Predefined delegate tooled to extract neural network
        topology parameters from a TensorFlow Sequential model.
        :param model: TensorFlow model of type Sequential.
        :param base: A Panda3D ShowBase object.
        :param neuron_path: The folder path were Level of Detail 3D mesh object are stored for neurons.
        :param edge_path: The folder path were Level of Detail 3D mesh object are stored for neurons.
        :param neuron_spacing: The amount of spacing between neurons on a single layer.
        :param layer_spacing: The amount of spacing between layers.
        :param alignment: The type of alignment used to align layers of holding different number of neurons.
        :param neuron_dimensions: The dimensions of the largest neuron.
        """

        super().__init__()

        self.base = base
        self.nnm = model

        # Network parameters
        self._neuron_spacing = neuron_spacing
        self._layer_spacing = layer_spacing
        self._alignment = alignment
        self._neuron_dimensions = neuron_dimensions

        # 3D mesh object file names
        self.ordered_neuron_mesh_paths = lod_3D_mesh_object_filenames(neuron_path, TopologyObject.NEURON)
        self.ordered_edge_mesh_paths = lod_3D_mesh_object_filenames(edge_path, TopologyObject.EDGE)

        # Extract model properties
        n_l, b_l = self._compute_neuron_counts()
        self.neurons_in_layer = n_l
        self.bias_in_layer = b_l

    """
    Neural Network Architecture Parameter Delegate Methods
    """

    def neuron_dimensions(self) -> LVecBase3f:
        return self._neuron_dimensions

    def level_of_details_neuron_nodes(self) -> List[NodePath]:
        # return a list of 3D mesh objects that defines
        # a neuron and different levels of detail.
        # From greatest to least level of detail
        return [self.base.loader.loadModel(filename) for filename in self.ordered_neuron_mesh_paths]

    def level_of_details_edge_nodes(self) -> List[NodePath]:
        # return a list of 3D mesh objects that defines
        # an edge and different levels of detail
        # From greatest to least level of detail
        return [self.base.loader.loadModel(filename) for filename in self.ordered_edge_mesh_paths]

    def number_of_neurons(self, for_layer: int) -> int:
        return self.neurons_in_layer[for_layer] + self.bias_in_layer[for_layer]

    def number_of_layers(self) -> int:
        return len(self.neurons_in_layer)

    def neuron_spacing(self) -> float:
        return self._neuron_spacing

    def layer_spacing(self) -> float:
        return self._layer_spacing

    def layer_alignment(self) -> TopologyAlignment:
        return self._alignment

    def connecting_neurons(self, layer_id: int, neuron_id: int) -> List[Tuple[int, int]]:
        return [(layer_id + 1, neuron) for neuron in range(self.neurons_in_layer[layer_id + 1])]

    """
    Architecture Helpers
    """

    def _compute_neuron_counts(self) -> Tuple[List[int], List[int]]:
        """
        Figure out the number of nodes to display
        per layer including the bias node.

        :return: A 2-tuple of lists where the list
        index is the layer and value is the node counts.
        """
        nodes_in_layer = []
        bias_in_layer = []
        node_weights = []

        for idx, layer in enumerate(self.nnm.layers):

            if not isinstance(layer, tf.keras.layers.Dense):
                continue

            node_weights = layer.weights[0].numpy()
            nodes_in_layer.append(np.prod(node_weights.shape[:-1]))

            if layer.bias is not None:
                bias_in_layer.append(1)
            else:
                bias_in_layer.append(0)

        nodes_in_layer.append(node_weights.shape[-1])
        bias_in_layer.append(0)

        return nodes_in_layer, bias_in_layer
