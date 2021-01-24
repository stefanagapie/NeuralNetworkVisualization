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
from math import tan

# Third party - Panda3D
from panda3d.core import deg2Rad, WindowProperties
from panda3d.core import NodePath, LVecBase3f
from panda3d.core import PointLight, AmbientLight
from direct.showbase.ShowBase import ShowBase

# Third party - TensorFlow
import tensorflow as tf
from tensorflow.keras import Sequential

# Packages
from NeuralNetworkTopology import NeuralNetworkSymmetricStratum
from TensorFlowTopology import TFSequentialModelSymmetricTopologyDelegate

if __name__ == '__main__':
    """
    Example of how to use the TFSequentialModelSymmetricTopologyDelegate class:
    
        1. Build and train a TensorFlow Sequential model.
        1. Instantiate a SymmetricTopologyDelegate (delegate), passing the model to it.
        3. Create an instance of NeuralNetworkSymmetricStratum (stratum), 
            passing along the delegate as a parameter.
        4. Call build() on the stratum.
        5. Create a subclass of ShowBase 
            (MyBasicNeuralNetworkApp -- a subclass of Panda3D game engine)
        6. Position the camera so we can see the Neural Network.
        7. Instantiate MyBasicNeuralNetworkApp(...) and call run() on this instance
            Optionally set flag 'lighting_and_animation=True' to true to apply lighting and animation.
    """


    class MyTensorFlowApp(ShowBase):

        def __init__(self, model: Sequential):
            super().__init__()

            # delegate extracts network parameters from tensorflow model
            lod_mesh_object_path = "assets_tensorflow"
            delegate = TFSequentialModelSymmetricTopologyDelegate(model, self, lod_mesh_object_path, lod_mesh_object_path)

            # builds neural network as defined by the delegate
            self.nns = NeuralNetworkSymmetricStratum(delegate)
            self.nns.build()

            # add network to the game scene
            self.nns.reparentTo(self.render)

            # self explanatory
            self.set_window_size(1100, 550)
            self.set_window_title("3D TensorFlow Visualization")
            self.center_camera_on_stratum(self.nns)

            # make it pretty
            self.add_light_source_to_node(self.cam)

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
            fudge = 103
            self.cam.setY(-distance + fudge)

        def set_window_size(self, width: float, height: float):
            wp = WindowProperties()
            wp.setSize(width, height)
            self.win.requestProperties(wp)

        def set_window_title(self, title: str):
            wp = WindowProperties()
            wp.setTitle(title)
            self.win.requestProperties(wp)

        """
        Animation & 3D Scene helpers
        """

        def add_light_source_to_node(self, light_source: NodePath):
            # place light source near camera
            plight = PointLight("plight")
            plight.setColor((0.945, 0.556, 0.203, 1))
            plight.setAttenuation((0.80, 0.001, 0))
            plnp = light_source.attachNewNode(plight)
            plnp.setPos(-20, 0, 0)
            self.render.setLight(plnp)

            # also set scene background color
            self.set_background_color(0.15, 0.15, 0.15, 1)

            # also add ambient lighting for effect
            alight = AmbientLight("alight")
            alight.setColor((0.24, 0.19, 0.16, 1))
            alnp = self.render.attachNewNode(alight)
            self.render.setLight(alnp)


    def tensorflow_sequential_model(epochs=1) -> Sequential:
        """Define and train a tensorflow model"""

        seed = 420
        tf.random.set_seed(seed)

        """
        Set up the model layers
        """
        initializer = tf.keras.initializers.he_normal(seed=seed)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(8, use_bias=False, activation=tf.nn.relu, kernel_initializer=initializer, bias_initializer=initializer),
            tf.keras.layers.Dropout(rate=0.01, seed=seed),
            tf.keras.layers.Dense(8, use_bias=False, activation=tf.nn.relu, kernel_initializer=initializer, bias_initializer=initializer),
            tf.keras.layers.Dense(4, use_bias=False, activation=tf.nn.sigmoid, kernel_initializer=initializer, bias_initializer=initializer)
        ])

        """
        Compile the model
        """
        model.compile(
            # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.06),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'],
        )

        """
        Training Data
        """
        X = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             # unknown
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        y = [[0, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 1, 1],
             [0, 1, 0, 0],
             [0, 1, 0, 1],
             [0, 1, 1, 0],
             [0, 1, 1, 1],
             [1, 0, 0, 0],
             [1, 0, 0, 1],
             # unknown
             [1, 0, 1, 0],
             [1, 0, 1, 1],
             [1, 1, 0, 0],
             [1, 1, 0, 1],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]

        """
        Train the model
        """
        model.fit(X, y, epochs=epochs, batch_size=2, shuffle=True, verbose=1)

        return model


    tensorflow_model = tensorflow_sequential_model(epochs=472)
    app = MyTensorFlowApp(tensorflow_model)
    app.run()
