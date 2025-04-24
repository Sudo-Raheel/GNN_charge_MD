# GNN_charge_MD

Graph Neural Network for charge-annotated Molecular Dynamics (MD) simulations.
Predicted charges from the GNN will be combined with the LJ potential to account for long range interaction. 
It is a modified approach to NN interatomic potential. Rather than directly training on the Forces and Energies, we are hoping for a boost in accuracy of the LJ potential. 
This approach has the following advantages:
1) Since the Charge Prediction using GNN has been show to work very accurately across different materials. This should act like a universal NN potential with minimal training.
2) Conventional NN potentials perform poorly outside the training regime. Verification and fine-tuning these potentials themselves is a data-intensive task. In the modified LJ approach, a single DFT snapshot and charge analysis can ascertain whether the DFT-level colonic interactions are being properly taken into account. 

## 📁 Repository Structure

- [`data/`](./data)  
  Contains input datasets, such as atomic structure files (e.g., CIF/POSCAR) and charge data.

- [`graph_build/`](./graph_build)  
  Utilities for constructing graphs from structures, including feature extraction for nodes (e.g., partial charges) and edges.
   Pymatgen ISSAEVNN class is used to create the adjacency matrix(bond table).

- [`utils/`](./utils)  
  Contains the files for featurization and scaling 

- [`results/`](./results)  
  Directory for storing trained models, evaluation outputs, visualizations, or logs.

- [`tests/bader/`](./tests/bader)  
  Contains tests and scripts related to Bader charge analysis and validation of charge parsing.

- [`__pycache__/`](./__pycache__)  
  Python bytecode cache.

## 📜 Scripts

- [`main.py`](./main.py)  
  Entry point script to run the training and evaluation pipeline for the GNN model.

- [`model_rh.py`](./model_rh.py)  
  Defines the model architecture and training logic for the GNN.

## 🚀 Getting Started

To use this repo:

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
