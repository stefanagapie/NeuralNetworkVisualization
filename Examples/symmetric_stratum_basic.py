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
from math import sin, tan

# Third party
from panda3d.core import deg2Rad
from panda3d.core import NodePath, LVecBase3f
from panda3d.core import PointLight, AmbientLight
from direct.showbase.ShowBase import ShowBase

# Packages
from NeuralNetworkTopology import TopologyAlignment
from NeuralNetworkTopology import SymmetricTopologyDelegate
from NeuralNetworkTopology import NeuralNetworkSymmetricStratum

if __name__ == '__main__':
    """
    Example of how to use the NeuralNetworkSymmetricStratum class:
    
        1. Implement a SymmetricTopologyDelegate (delegate).
        2. Instantiate the delegate.
        3. Create an instance of NeuralNetworkSymmetricStratum (stratum), 
            passing along the delegate as a parameter.
        4. Call build() on the stratum.
        5. Create a subclass of ShowBase 
            (MyBasicNeuralNetworkApp -- a subclass of Panda3D game engine)
        6. Position the camera so we can see the Neural Network.
        7. Instantiate MyBasicNeuralNetworkApp(...) and call run() on this instance
            Optionally set flag 'lighting_and_animation=True' to true to apply lighting and animation.
    """


    class MyDelegate(SymmetricTopologyDelegate):

        """
        The methods below define the topology of our neural network;
        they are called by the NeuralNetworkSymmetricStratum instance
        after the build() method is invoked.
        """

        def __init__(self, base: ShowBase):
            self.base = base

            # The where we define the number of network
            # layers and the number of neurons per layer.
            # >> MODIFY ME << to see different network topologies.
            self.neurons_per_layer = [8, 16, 4, 8, 4]

        def neuron_dimensions(self) -> LVecBase3f:
            return LVecBase3f(1, 1, 1)

        def level_of_details_neuron_nodes(self) -> List[NodePath]:
            # return a list of 3D mesh objects that defines
            # a neuron and different levels of detail.
            # From greatest to least level of detail
            return [self.base.loader.loadModel(f"assets/neuron_1280T.obj")]

        def level_of_details_edge_nodes(self) -> List[NodePath]:
            # return a list of 3D mesh objects that defines
            # an edge and different levels of detail
            # From greatest to least level of detail
            return [self.base.loader.loadModel(f"assets/cylinder_172T.obj")]

        def number_of_layers(self) -> int:
            return len(self.neurons_per_layer)

        def number_of_neurons(self, for_layer: int) -> int:
            return self.neurons_per_layer[for_layer]

        def neuron_spacing(self) -> float:
            return 2.5

        def layer_spacing(self) -> float:
            return 16

        def layer_alignment(self) -> TopologyAlignment:
            return TopologyAlignment.CENTER

        def connecting_neurons(self, layer_id: int, neuron_id: int) -> List[Tuple[int, int]]:
            return [(layer_id + 1, neuron) for neuron in range(self.neurons_per_layer[layer_id + 1])]


    class MyBasicNeuralNetworkApp(ShowBase):

        def __init__(self, lighting_and_animation: bool):
            super().__init__()

            delegate = MyDelegate(self)
            self.stratum = NeuralNetworkSymmetricStratum(delegate)
            self.stratum.build()

            if not lighting_and_animation:
                """
                Setup without scene lighting or animation
                """
                # add network stratum to root of game scene
                self.stratum.reparentTo(self.render)

                # self explanatory
                self.center_camera_on_stratum(self.stratum)
            else:
                """
                Setup with scene lighting and animation
                """
                # create a pivot node in order to
                # define the place of stratum rotation
                self.center_pivot_node = self.render.attachNewNode("center_pivot_node")
                self.center_pivot_node.reparentTo(self.render)
                self.stratum.reparentTo(self.center_pivot_node)

                # self explanatory
                self.center_stratum_on_pivot()
                self.center_camera_on_stratum(self.center_pivot_node)
                self.add_light_source_to_node(self.cam)

                # begin rotation task
                self.taskMgr.add(self.rotate_network, "rotate-network")

        def center_camera_on_stratum(self, stratum: NodePath):
            # place camera on center of the network
            bounds = stratum.getTightBounds()
            center = (bounds[0] + bounds[1]) / 2
            self.cam.setPos(center)

            # zoom the camera out to view entire network
            fov = self.camLens.getFov()
            x_length = bounds[1].x - bounds[0].x
            z_length = bounds[1].z - bounds[0].z
            distance = max(x_length, z_length) / tan(deg2Rad(min(fov[0], fov[1])))
            self.cam.setY(-distance)

        """
        Animation helpers
        """

        def add_light_source_to_node(self, light_source: NodePath):

            # place light source near camera
            plight = PointLight("plight")
            plight.setColor((0, 0.75, 1, 1))
            plight.setAttenuation((1, 0.001, 0))
            plnp = light_source.attachNewNode(plight)
            plnp.setPos(-20, 0, 0)
            self.render.setLight(plnp)

            # also set scene background color
            self.set_background_color(0.15, 0.15, 0.15, 1)

            # also add ambient lighting for effect
            alight = AmbientLight("alight")
            alight.setColor((0.14, 0.14, 0.24, 1))
            alnp = self.render.attachNewNode(alight)
            self.render.setLight(alnp)

        def center_stratum_on_pivot(self):
            # since the pivot node is in the lower
            # left corner we place the center of the
            # stratum on that pivot node
            bounds = self.stratum.getTightBounds()
            center = -(bounds[0] + bounds[1]) / 2
            self.stratum.setPos(center)

        def rotate_network(self, task):
            c_pos = self.center_pivot_node.getPos()

            # rotate the stratum about the z-axis
            self.center_pivot_node.setH(task.time * 20)
            self.center_pivot_node.setPos(c_pos.x, sin(task.time / 2) * 45 - 25, c_pos.z)
            return task.cont


    app = MyBasicNeuralNetworkApp(lighting_and_animation=False)
    app.run()
