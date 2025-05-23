# DFU Classification with Graph Convolutional Networks (GCN)
This project implements a Graph Convolutional Network (GCN) using PyTorch Geometric and PyTorch Lightning to classify images as Diabetic Foot Ulcer (DFU) or Non-DFU. The approach involves preprocessing images into graph structures using superpixel segmentation (SLIC) and Region Adjacency Graphs (RAG), followed by training a GCN model for binary classification.
Dataset Link is - https://universe.roboflow.com/ulcer-detection/foot-ulcer-detection/dataset/2
                -  https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu
## Overview
The pipeline preprocesses images by:
- Resizing images to a fixed size (264x264).
- Segmenting them into superpixels using the SLIC algorithm.
- Creating a graph representation with nodes (superpixels) and edges (adjacency relationships).
- Extracting features such as mean color, centroid positions, area, and eccentricity for each node.
- Training a GCN model to classify the graphs as DFU (0) or Non-DFU (1).
- The model is trained using PyTorch Lightning with early stopping and checkpointing, and predictions are visualized with superpixel clusters and graph overlays.

## Requirements
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- PyTorch Lightning
- Torchvision
- PIL (Python Imaging Library)
- NumPy
- Scikit-Image
- Scikit-Learn
- Seaborn
- Matplotlib
- NetworkX

## Code Structure
- **Imports**: Libraries for image processing, graph handling, and model training.
- **Preprocessing**:  
    - `create_graph`: Converts an image into a graph with node features and edges.  
    - `load_dataset`: Loads and balances the dataset.  
- **Prediction**:  
    - `predict_image`: Runs inference on a single image and optionally visualizes the results.  
- **Constants**:  
    - `IMAGE_SIZE`: (264, 264)  
    - `SLIC_PARAMS`: Parameters for SLIC segmentation (`n_segments=25`, `compactness=10`, `sigma=1.0`)  
- `DATASET_PATH`: Path to the dataset directory.

# Notes

## Model Definition
The `DFUGCN` class is not included in the provided snippet. It should:
- Extend `pl.LightningModule`
- Use `GCNConv` layers for graph processing

## Visualization
- Requires a graphical environment (e.g., Jupyter Notebook)

## Tuning
- Adjust `SLIC_PARAMS` (e.g., `n_segments`) for optimal segmentation based on your images

## Checkpoint
- Ensure `best_model_path` matches your saved checkpoint location

## Contributing
Feel free to open issues or submit pull requests with:
- Improvements
- Bug fixes
- New features
