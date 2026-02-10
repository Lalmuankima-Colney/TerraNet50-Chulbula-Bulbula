# TerraNet50-Chulbula-Bulbula

This repository contains the AI model developed by **Team Chulbula Bulbula** for the **TerraNet50 Hackathon**.

The project implements a deep learning–based **semantic segmentation** system for rugged off-road terrain images using a **U-Net–style convolutional neural network** trained with **TensorFlow**.

The work is based on the **Terra Seg: Rugged Terrain Segmentation** dataset from **Kaggle** and was developed and trained using **Google Colab**.

---

## Project Overview

Autonomous vehicles and off-road robotic systems require an accurate understanding of terrain structure to navigate safely.
This project focuses on **pixel-level classification** to distinguish between:

* **Traversable terrain**
* **Non-traversable terrain**

**Inputs and Outputs**:

* **Input**: RGB terrain images
* **Output**: Binary semantic segmentation masks

---

## Dataset

* **Source**: Kaggle — *Terra Seg: Rugged Terrain Segmentation*
* **Images**: RGB off-road terrain images
* **Masks**: Binary ground-truth segmentation masks

> ⚠️ **Note**: Due to GitHub file size limitations, the full dataset and certain large files are **not included** directly in this repository.

---

## Model Architecture

* **Architecture**: U-Net (via `segmentation_models`)
* **Encoder / Backbone**: SE-ResNeXt50 (`seresnext50`, ImageNet pretrained)
* **Framework**: TensorFlow / Keras
* **Decoder**: Transposed convolution blocks
* **Output Activation**: Sigmoid (binary segmentation)

---

## Data Preprocessing

* Images read using OpenCV and converted from BGR to RGB
* Images resized to **544 × 960** resolution
* Images normalized using backbone-specific preprocessing
* Masks resized using nearest-neighbor interpolation
* Binary mask creation using class labels **27** and **39**
* Masks reshaped to `(H, W, 1)` format

---

## Dataset Setup

This project is designed to run on **Kaggle Notebooks** or **Google Colab**.

### Kaggle Directory Structure

```
/kaggle/input/terra-seg-rugged-terrain-segmentation/
└── offroad-seg-kaggle/
    ├── train_images/
    ├── train_masks/
    └── test_images_padded/
```

### Steps

1. Add the **Terra Seg: Rugged Terrain Segmentation** dataset to your Kaggle notebook.
2. Ensure the dataset is available at the path shown above.
3. Run the training notebook to start K-Fold training.

---

## Files and Downloads

A **Google Drive link** is provided to download large files that cannot be hosted on GitHub.

Due to **GitHub size limitations**, the following files are **not included** in this repository:

* Dataset CSV files
* Trained model weight files ( only **Model 5**)

### How to use the Drive files

1. Open the provided **Google Drive link**.
2. Download the entire folder or only the required files.
3. Place the downloaded files inside the `Model/` directory of this project.

---

## Training Strategy

* **Cross-Validation**: 5-Fold K-Fold Cross Validation
* **Batch Size**: 4
* **Epochs**: 7 per fold
* **Optimizer**: Adam (LR = 1e-4)
* **Loss Function**: Binary Cross Entropy + Dice Loss
* **Metric**: IoU Score (threshold = 0.48)
* **Learning Rate Schedule**: Cosine Annealing
* **Checkpointing**: Best model saved per fold based on validation IoU
* **Post-training**: Validation threshold tuning (0.3 – 0.7) per fold
* Add test-time augmentation (TTA)
* Ensemble K-Fold models

---

## Data Augmentation

Augmentations applied using **Albumentations**, optimized for off-road terrain:

* Horizontal Flip
* Shift, Scale, and Rotation
* Random Brightness & Contrast
* RGB Shift (robust to lighting changes)
* Coarse Dropout (simulates occlusions)

---

## Future Work

* Train for additional epochs
* Experiment with larger backbones (EfficientNet, ResNet101)
* Evaluate using Dice Score and Precision–Recall metrics
* Visualize predictions on unseen test images

---

## Acknowledgements

* Kaggle for the Terra Seg dataset
* `segmentation_models` library
* Google Colab and Kaggle for compute resources
* TerraNet50 Hackathon organizers

by Ayush Nandapure, Lalmuankima Colney, Pratik Phophaliya
