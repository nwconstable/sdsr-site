# Project: Networked Edge Training for Spatial Graph Learning
## Group 2 - Noah Constable, Joe Conroy

### Purpose
The purpose of this project is to train a network of simulated edge devices (i.e. drones) using multiple different techniques including federated average learning and gossip learning. We want to empirically characterize each of these methods against each other to determine approximate bounds on resources such as network connectivity and compute. 

### Resources
One of the data sets we are using is the [Minnesota National Wetlands Inventory Update](https://www.dnr.state.mn.us/wetlands/nwi_proj.html) GeoPackage. This is a GIS dataset that requires 'wrangling' into an appropriate form for our purposes.

To use the programs, especially any of the PyG scripts, a CUDA-enabled GPU is recommended for performance but **not required**. All scripts should run on a CPU-only installation of PyTorch and PyTorch Geometric. If you have a compatible NVIDIA GPU and want GPU acceleration, follow the [official CUDA installation guide](https://developer.nvidia.com/cuda-toolkit) and then install a CUDA-enabled build of [PyTorch](https://pytorch.org/get-started/locally/). For CPU-only use, install the CPU-only PyTorch wheels as described in the PyTorch documentation.

### How to Use
1. (Optional) [Enable Python venv](https://www.w3schools.com/python/python_virtualenv.asp)
2. Install [dependencies](#libraries)
3. Using Python, run the scripts in the following order (unless you know what you're doing):  
    a. `load_data.py`  
    b. `build_graph.py`  

### Libraries
Packages listed here can be installed by using:

`pip install {package_name}`

Python libraries used include:
- geopandas
- shapely (included with geopandas)
- numpy
- torch_geometric
- torch