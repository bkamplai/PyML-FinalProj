import cv2 # For reading in images
import torch # PyTorch library
import torch.nn as nn # Neural network
import torch.optim as optim # For optimization algorithms
from torch.utils.data import Dataset, DataLoader # To use pre-loaded dataset
import torchvision.transforms as transforms # To transform images from dataset
import torchvision models as models # Models for image, video classification, etc.
import os # To save model to disk

import matplotlib.pyplot as plt # For plotting data
import numpy as np # For math on tensors
import pandas as pd # For reading csv files

"""
ASL Alphabet Fingerspelling Classifier:
This program aims to develop a machine learning-based classifier for ASL fingerspelling recognition 
using CNNs and image augmentation. The system will enable real-time recognition and feedback for ASL
education and accessibility.

Authors: Brandon Kamplain, Aidan Kavanagh, Elena Schmitt
"""

# Enter values for mean and std dev depending on model chosen
model_mean = []
model_std = []

# Transform for images to keep consistent
transform = transforms.Compose([
    transforms.Resize(512, 512), # All images in the dataset are 512x512
    transforms.ToTensor(), # Convert the image to a tensor for torch
    transforms.Normalize(mean=model_mean, std=model_std) # Mean and std dev for model
])


"""
Dataset class for ASL alphabet images dataset.
"""
class ASLDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass


"""
Run the train data through a model and create/save the trained model to disk.
"""
def train():
    pass


"""
Run the test data through the model and return the accuracy percentage.
"""
def test():
    pass


def main() -> None:
    pass



if __name__ == "__main__":
    main()
