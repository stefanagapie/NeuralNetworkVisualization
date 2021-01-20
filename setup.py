import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NeuralNetworkVisualization-stefanagapie",
    version="0.0.1",
    author="Stefan Agapie",
    author_email="stefanagapie@gmail.com",
    description="APIs for 3D Neural Network Visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanagapie/NeuralNetworkVisualization",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)