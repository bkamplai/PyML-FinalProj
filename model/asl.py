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
Takes in an image, runs it through the model, and returns what letter
the model classified it as.

asl_fingerspell_mobilenet_finetuned.keras
"""
def classify(image):
    pass


"""
Main function where 
"""
def main() -> None:
    # Name of saved model.
    model_save = 'asl_model.pth'

    # Check if model exists
    if (os.path.exists(model_save)):
        pass
    else:
        print("Model does not exist.\n")


if __name__ == "__main__":
    main()