# GNN_charge_MD

Graph Neural Network for charge-annotated Molecular Dynamics (MD) simulations. This repository contains the pipeline for building charge-decorated graphs from atomic structures and training GNN models to predict charge-related properties.

## 📁 Repository Structure

- [`data/`](./data)  
  Contains input datasets, such as atomic structure files (e.g., CIF/POSCAR) and charge data.

- [`graph_build/`](./graph_build)  
  Utilities for constructing graphs from structures, including feature extraction for nodes (e.g., partial charges) and edges (e.g., inverse distances).

- [`utils/`](./utils)  
  Helper functions and modules for preprocessing, configuration, and general utilities used across the project.

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
