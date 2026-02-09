# TerraNet50-Chulbula-Bulbula
This is a repository for AI model made by the team Chulbula Bulbula for the TerraNet50 hackathon 
This project implements a deep learning model for semantic segmentation of rugged off-road terrain images using a U-Net–style convolutional neural network trained with PyTorch.

The project is based on the Terra Seg: Rugged Terrain Segmentation dataset from Kaggle and was developed using Google Colab.

---

## Project Overview

Autonomous vehicles and off-road robotic systems require an understanding of terrain structure to navigate safely.  
This project focuses on pixel-level classification to distinguish traversable and non-traversable terrain regions.

- Input: RGB terrain images  
- Output: Binary segmentation masks

---

## Dataset

- Source: Kaggle — Terra Seg: Rugged Terrain Segmentation
- Images: RGB off-road terrain images
- Masks: Binary segmentation ground truth

## Model Architecture

- Architecture: Lightweight U-Net–style Convolutional Neural Network
- Framework: PyTorch
- Loss Function: Binary Cross Entropy (BCE)
- Optimizer: Adam
- Activation Functions: ReLU, Sigmoid

---

## Data Preprocessing

- Images converted to float32 format
- Pixel values normalized to the range [0, 1]
- Images reshaped to (Channels, Height, Width)
- Masks scaled and converted to binary tensors

---

### Dataset Setup
1. Download the Terra Seg dataset from Kaggle.
2. Upload the dataset ZIP file to Google Drive.
3. Extract the dataset into the following directory:/content/offroad-seg-kaggle/

## Future Work

- Implement a deeper U-Net architecture
- Add data augmentation techniques
- Train for additional epochs
- Evaluate performance using IoU and Dice metrics
- Visualize model predictions on test images

---
