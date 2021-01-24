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
from typing import List, Tuple, Dict
from collections import OrderedDict
from math import inf

# Third party
from panda3d.core import NodePath, LODNode
from panda3d.core import LVector3f

# Package
from NeuralNetworkTopology import SymmetricTopologyDelegate
from NeuralNetworkTopology import TopologyAlignment


class NeuralNetworkSymmetricStratum(NodePath):

    # TODO: Description please

    def __init__(self, delegate: SymmetricTopologyDelegate):
        super().__init__("stratum")

        self.delegate: SymmetricTopologyDelegate = delegate

    @staticmethod
    def neuron_tag(layer_id: int, neuron_id: int):
        return f"(Neuron): Layer_id={layer_id}, Neuron_id={neuron_id}"

    @staticmethod
    def edge_tag(source_layer_id: int, source_neuron_id: int, target_layer_id: int, target_neuron_id: int):
        return f"(Edge): " \
               f"SourceLayer_id={source_layer_id}, SourceNeuron_id={source_neuron_id}, " \
               f"TargetLayer_id={target_layer_id}, TargetNeuron_id={target_neuron_id}"

    def build(self):
        max_neurons, layer_data = self._max_and_neurons_per_layer()
        self._layout_neurons(max_neurons, layer_data)
        self._layout_edges()

    def _max_and_neurons_per_layer(self) -> Tuple[int, Dict[int, int]]:
        """
        Stores the number of neurons per layer and notes the
        largest number of neurons in an individual layer.
        :return:
            A 2-tuple where the first value is the largest
            amount observed neurons for an individual layer
            and the second value is a list of tuples with the
            following form [(layer, number_of_neurons),...]
        """

        max_neurons_for_layer = 0
        layer_to_neurons_map = OrderedDict()

        for layer in range(self.delegate.number_of_layers()):
            # iterate over the number of neural network layers

            # request number of neurons for some layer i
            neurons = self.delegate.number_of_neurons(layer)

            layer_to_neurons_map[layer] = neurons

            # capture maximum number of neurons for an individual lary
            max_neurons_for_layer = max(max_neurons_for_layer, neurons)

        return max_neurons_for_layer, layer_to_neurons_map

    def _layout_edges(self):

        for source_layer_id in range(self.delegate.number_of_layers() - 1):
            for source_neuron_id in range(self.delegate.number_of_neurons(source_layer_id)):
                """
                These first two for loops generate the source neurons identifying parameters
                """

                # TODO: figure out what the running time is for this find API.
                #  Use a local map if it's not constant time
                source_node: NodePath = self.find(f"**/{self.neuron_tag(source_layer_id, source_neuron_id)}")

                for target_layer_id, target_neuron_id in self.delegate.connecting_neurons(source_layer_id, source_neuron_id):
                    """
                    First request the connecting neurons from the architecting delegate, then
                    connect the source neuron to the provided target neurons
                    """

                    lod_tag = self.edge_tag(source_layer_id, source_neuron_id, target_layer_id, target_neuron_id)
                    lod = LODNode(lod_tag)
                    lod_np = NodePath(lod)
                    lod_np.setScale(0.05, 1, 0.05)  # TODO: allow for customization of diameter to delegate
                    lod_np.reparentTo(self)

                    # TODO: figure out what the running time is for this find API.
                    #  Use a local map if it's not constant time
                    target_node: NodePath = self.find(f"**/{self.neuron_tag(target_layer_id, target_neuron_id)}")

                    # compute the midpoint between source and target
                    # the connecting edge is placed on this point
                    mid_pos = (source_node.getPos() + target_node.getPos()) / 2

                    # compute the distance between source and target
                    # this distance is used to scale the edge along the y-axis
                    # divide distance by two since edge origin is midpoint
                    edge_scale_y = source_node.getDistance(target_node) / 2

                    # adjust position so that the edge lies
                    # between source node and target node
                    lod_np.setPos(mid_pos)

                    # adjust scale so that the edge
                    # touches both source and target nodes
                    current_scale = lod_np.getScale()
                    lod_np.setScale(current_scale.x, edge_scale_y, current_scale.z)

                    # have the y-axis of the edge point to the target node
                    lod_np.lookAt(target_node)

                    # TODO: the code below involving LOD is duplicated in the _layout_neurons(..) method (Consolidate)

                    ratio = 1.8  # switch distance ratio between successive neuron models
                    switch = (23, 0)  # first LOD switch range assignment for model with greatest details

                    # request level of details neuron models
                    lod_edges = self.delegate.level_of_details_edge_nodes()

                    for idx, neuron in enumerate(lod_edges):
                        """
                        LOD: Level of Detail
                        Select the models to be used at various levels of detail, from models to camera,
                        assigning the model with the least triangles to the farthest distance range
                        """

                        # the last switch occurs at infinity
                        # (furthest neuron is always visible)
                        if idx == len(lod_edges) - 1:
                            switch = (inf, switch[1])

                        # apply neuron model to switch range
                        lod.addSwitch(*switch)
                        neuron.reparentTo(lod_np)

                        # compute the next switch range
                        upper = switch[0] * ratio
                        lower = switch[0]
                        switch = (upper, lower)

    def _neuron_positions_on_xz_plane(self, max_neurons: int, layer_to_neuron_map: Dict[int, int]) -> List[List[LVector3f]]:

        alignment = self.delegate.layer_alignment()

        # request typical neuron dimensions
        x_dim, z_dim, y_dim = self.delegate.neuron_dimensions()

        # request neuron and layer spacing and compute center offsets
        neuron_spacing = self.delegate.neuron_spacing()
        layer_spacing = self.delegate.layer_spacing()
        neuron_offset = z_dim + neuron_spacing
        layer_offset = x_dim + layer_spacing

        # compute largest network width. we define the width as to be the largest
        # distance across an individual layer of neurons computed from center to
        # center of neuron extremes
        largest_width = (max_neurons - 1) * neuron_offset

        # initialize neuron position matrix
        neuron_positions = [[] for _ in range(len(layer_to_neuron_map))]

        for layer_i, neurons in layer_to_neuron_map.items():

            # recompute neuron offset since this may change
            # depending on the type of layer alignment selected
            # TODO: this can be improved to avoid this code duplication
            neuron_offset = z_dim + neuron_spacing

            # compute network width of layer i
            current_width = (neurons - 1) * neuron_offset

            if alignment == TopologyAlignment.CENTER:

                # compute a centering offset such that the layers with the smaller width
                # are centered on the layers with the largest width
                centering_offset = (largest_width - current_width) / 2

            elif alignment == TopologyAlignment.JUSTIFIED:

                centering_offset = 0

                if neurons == 1:
                    # if there is only one neuron then center it
                    centering_offset = largest_width / 2

                elif current_width < largest_width:
                    neuron_offset = largest_width / (neurons - 1)

            else:
                raise ValueError("Invalid Layer Alignment " + str(alignment))

            for neuron_j in range(neurons):
                # compute position of neuron
                neuron_pos = LVector3f(layer_i * layer_offset, 0, (neuron_j * neuron_offset) + centering_offset)

                neuron_positions[layer_i].append(neuron_pos)

        return neuron_positions

    def _layout_neurons(self, max_neurons: int, layer_to_neurons_map: Dict[int, int]):

        neuron_positions = self._neuron_positions_on_xz_plane(max_neurons, layer_to_neurons_map)

        for layer_i, neurons in layer_to_neurons_map.items():
            """
            Create a stratum of neurons
            """

            for neuron_j in range(neurons):
                """
                Create a layer of neurons
                """

                neuron_pos = neuron_positions[layer_i][neuron_j]

                lod_tag = self.neuron_tag(layer_i, neuron_j)
                lod = LODNode(lod_tag)
                lod_np = NodePath(lod)
                lod_np.setPos(neuron_pos)
                lod_np.reparentTo(self)

                # TODO: the code below involving LOD is duplicated in the _layout_edges(..) method (Consolidate)

                ratio = 1.8  # switch distance ratio between successive neuron models
                switch = (23, 0)  # first LOD switch range assignment for model with greatest details

                # request level of details neuron models
                lod_neurons = self.delegate.level_of_details_neuron_nodes()

                for idx, neuron in enumerate(lod_neurons):
                    """
                    LOD: Level of Detail
                    Select the models to be used at various levels of detail, determined by 
                    distance from camera to model, assigning the model with the least triangles 
                    to the farthest distance range
                    """

                    # the last switch occurs at infinity
                    # (furthest neuron is always visible)
                    if idx == len(lod_neurons) - 1:
                        switch = (inf, switch[1])

                    # apply neuron model to switch range
                    lod.addSwitch(*switch)
                    neuron.reparentTo(lod_np)

                    # compute the next switch range
                    upper = switch[0] * ratio
                    lower = switch[0]
                    switch = (upper, lower)
