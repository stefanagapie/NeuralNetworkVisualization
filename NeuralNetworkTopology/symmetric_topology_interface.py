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
from enum import Enum, auto
from typing import List, Tuple
from abc import ABC, abstractmethod

# Third party
from panda3d.core import NodePath
from panda3d.core import LVecBase3f


class TopologyAlignment(Enum):
    CENTER = auto()
    JUSTIFIED = auto()


class SymmetricTopologyDelegate(ABC):

    # TODO: Description please

    @abstractmethod
    def neuron_dimensions(self) -> LVecBase3f:
        """
        Return the dimensions of the largest neuron model
        """
        raise NotImplementedError()

    @abstractmethod
    def level_of_details_neuron_nodes(self) -> List[NodePath]:
        """
        Return a list of neuron model nodes from
        greatest level of detail ot least level of detail.

        Greatest level of detail is a model node with the
        most amount of mesh triangles.
        """
        raise NotImplementedError()

    @abstractmethod
    def level_of_details_edge_nodes(self) -> List[NodePath]:
        """
        Return a list of edge model nodes from
        greatest level of detail to least level of detail.

        Greatest level of detail is a model node with the
        most amount of mesh triangles.
        """
        raise NotImplementedError()

    @abstractmethod
    def number_of_layers(self) -> int:
        """
        Return the number of neural network layers.
        """
        raise NotImplementedError()

    @abstractmethod
    def number_of_neurons(self, for_layer: int) -> int:
        """
        Return the number of neurons for a given layer
        """
        raise NotImplementedError()

    @abstractmethod
    def neuron_spacing(self) -> float:
        # TODO: Allow for neuron spacing per layer in CUSTOM mode.
        """
        Return the spacing between neurons for each layer.
        """
        raise NotImplementedError()

    @abstractmethod
    def layer_spacing(self) -> float:
        # TODO: Allow layer spacing per layer in CUSTOM mode.
        """
        Return the spacing between layers.
        """
        raise NotImplementedError()

    @abstractmethod
    def layer_alignment(self) -> TopologyAlignment:
        """
        Return the desired alignment for layers
        that have an unequal number of neurons.
        """
        raise NotImplementedError()

    @abstractmethod
    def connecting_neurons(self, layer_id: int, neuron_id: int) -> List[Tuple[int, int]]:
        """
        Given a neuron_id and a layer_id you are asked
        which neurons connect to the given neuron.
        :param layer_id: The layer id that the given neuron belongs to.
        :param neuron_id: The neuron id that is positioned in the given layer.
        :return: A list ot 2-tuple where each tuple identifies a target neuron
        that is to be connected to the given neuron. The first value in a tuple,
        index 0, identifies the target layer. The second value in a tuple, index
        1, identifies the target neuron within a target layer.
        """
        raise NotImplementedError()
