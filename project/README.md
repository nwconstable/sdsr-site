# Project: Networked Edge Training for Spatial Graph Learning
## Group 2 - Noah Constable, Joe Conroy

### Purpose
The purpose of this project is to train a network of simulated edge devices (i.e. drones) using multiple different techniques including federated average learning and gossip learning. We want to empirically characterize each of these methods against each other to determine approxiamate bounds on resources such as network connectivity and compute. 

### Resources
One of the data sets we are using is the [Minnesota National Wetlands Inventory Update](https://www.dnr.state.mn.us/wetlands/nwi_proj.html) GeoPackage. This is a GIS dataset that requires 'wrangling' into an appropriate form for our purposes.

To use the programs, especially any of the PyG scripts, you will need CUDA enabled on your machine. You can find instructions on how to install CUDA on supported machine architectures. This SDK is directly used by PyTorch libraries, and is necessary to specify when installing [torch](https://developer.nvidia.com/cuda/toolkit).

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