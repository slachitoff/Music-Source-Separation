# Music Source Separation with U-Net

This repository contains the resources and links for a deep learning project focused on separating music tracks into their constituent sources, such as vocals, bass, drums, and other instruments, using a U-Net architecture.

## Repository Contents

- A direct link to the Google Colab notebook used for the project.
- Precomputed spectrograms stored as `.pt` files in the repository for training and evaluation.
- Links to the HDF5 files for training and test spectrograms stored on Google Drive.
- The `model_parameters.pt` file containing the saved parameters from our most successful training run.

## Project Overview

The goal of this project is to explore music source separation techniques using a U-Net architecture adapted for complex spectrograms.

## Model Overview

The `model_parameters.pt` file is a PyTorch checkpoint containing the weights of our trained U-Net model. This model has been fine-tuned to separate music tracks into vocals, bass, drums, and other instruments.

## Getting Started

### Google Colab Notebook

To access and interact with the project notebook:
1. Follow the provided link to the Google Colab environment: [Music Source Separation](https://colab.research.google.com/github/slachitoff/Music-Source-Separation/blob/main/Music_Source_Separation_with_U_Net.ipynb)
2. The notebook can be run in your browser, allowing you to replicate our results or further experiment with the model.

### Downloading the Spectrogram Files

The precomputed spectrograms are stored both as `.pt` files in this repository and as HDF5 files on Google Drive.

- **Training Spectrograms (.pt)**: `Precomputed Spectrograms/Training Spectrograms`
- **Test Spectrograms (.pt)**: `Precomputed Spectrograms/Test Spectrograms`
- **Training Spectrograms (HDF5)**: [Download from Google Drive](https://drive.google.com/file/d/1lqcjyp_pqqXxL14vOul9_RDzFZfl5TdP/view)
- **Test Spectrograms (HDF5)**: [Download from Google Drive](https://drive.google.com/file/d/1LcicQ3QW9pP4U9X-0yISzOPETNubWoFh/view)

After downloading the HDF5 files, place them in the appropriate directory.

### Using the Pre-Trained Model

To utilize the pre-trained model (`model_parameters.pt`), ensure you have PyTorch installed and then follow these steps:

1. Download the `model_parameters.pt` file from this repository.
2. Use the following code snippet in your PyTorch environment to load the model:

```python
import torch
from model import EnhancedUNet  # Ensure you have the model architecture defined or imported

# Set the device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize an instance of the EnhancedUNet model
model = EnhancedUNet(in_channels=2, num_sources=4, dropout_prob=0.3)

# Load the pretrained model from a file
state_dict = torch.load('path/to/model_parameters.pt', map_location=device)
model.load_state_dict(state_dict)

# Move the model to the designated computing device
model.to(device)
model.eval()  # Switch the model to evaluation mode
```

### Evaluation Results

These metrics were averaged across the MUSDB18 test set:

| Source  |   SDR   |   SIR   |   SAR   |   ISR   |
|---------|--------:|--------:|--------:|--------:|
| Bass    |   0.11  | -14.92  |   5.73  |   0.15  |
| Drums   |   0.85  |  -5.74  |   7.74  |   1.29  |
| Other   |   1.66  |  -0.78  |   7.69  |   2.64  |
| Vocals  |   0.90  |  -5.51  |   7.65  |   1.47  |

