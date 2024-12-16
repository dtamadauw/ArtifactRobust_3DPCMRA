# Pulsation Artifacts Augmentation for DL-Based Segmentation

This repository contains **source code** and **trained models** for augmenting pulsation artifacts in medical images to improve deep learning-based segmentation. The primary model used is the **3D U-Net**, which is publicly available but not included in this repository due to licensing reasons. Please download the 3D U-Net source code from the original page: [3DUnetCNN](https://github.com/ellisdg/3DUnetCNN).

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Trained Model](#trained-model)
- [Acknowledgments](#acknowledgments)

## Introduction

Pulsation artifacts can significantly affect the quality of medical image segmentation. This project provides tools to **augment pulsation artifacts** in datasets to train more robust segmentation models. By simulating these artifacts during training, the model becomes more resilient to real-world imperfections in medical imaging.

The code has been modified from the original, as the original utilized GE's proprietary library. This version utilizes a magnitude NRRD input rather than a complex array.


## Requirements

- **Python 3**
- **PyTorch**
- **NumPy**
- **[3DUnetCNN](https://github.com/ellisdg/3DUnetCNN)**

## Installation


1. **Clone this repository:**

   ```git clone https://github.com/dtamadauw/ArtifactRobust_3DPCMRA```

2. **Install required Python packages:**

   ```pip install -r requirements.txt```


3. **Download the 3D U-Net source code:**

   Download and set up the 3D U-Net from the [original repository](https://github.com/ellisdg/3DUnetCNN).

## Usage

**Data Augmentation**

    Augmentation.py is the primary script used for augmenting pulsation artifacts in your dataset.

   ```python Augmentation.py [Training data directory] [Output directory]```

    Generating Config File
    Before training the 3D U-Net model, you need to generate a configuration file using Gen_config_file.py.


   ```python Gen_config_file.py```

    Segmentation Prediction
    Use predict.py to perform segmentation using the trained 3D U-Net model.
    bash

   ```python predict.py --config_filename [Path to generated configuration file]] --model_filename [Path to trained] --group [Group Name] --output_directory [Directory for segmentation results] --input_dir [Directory for input MRA data]```



## Trained Model

    The trained model is available and can be used directly with predict.py.
    Model Path: Trained/trained_model.pth

## Acknowledgments

    The implementation of the 3D U-Net model is based on the work from 3DUnetCNN.
    Note: Since the license of the 3D U-Net network is not clear, we have not included its source code in this repository. Please ensure that you comply with all licensing requirements when using the 3D U-Net code.
