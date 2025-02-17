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
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (resnet values)
model_mean = [0.485, 0.456, 0.406]
model_std = [0.229, 0.224, 0.225]

# Transform for images to keep consistent
transform = transforms.Compose([
    transforms.ToPILImage(), # Convert image to PIL to be able to use ToTensor()
    transforms.Resize(512, 512), # All images in the dataset are 512x512
    transforms.ToTensor(), # Convert the image to a tensor for torch
    transforms.Normalize(mean=model_mean, std=model_std) # Mean and std dev for model
])


"""
Dataset class for ASL alphabet images dataset.
"""
class ASLDataset(Dataset):
    def __init__(self, file, directory, transform=None):
        self.data = pd.read_csv(file) # Use pandas to read csv
        self.directory = directory # Directory of image
        self.transform = transform # Apply transform to image
        self.letter = self.data.columns[1:].tolist() # Get letter image represents from csv
    
    def __len__(self):
        return len(self.data) # Return length of data

    def __getitem__(self, idx):
        image_name = os.path.join(self.directory, self.data.iloc[idx, 0]) # Gets filename of image
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert image from BGR to RGB
        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(np.float32)) # Get labels from csv and conver to float

        if self.transform:
            image = self.transform(image)
        
        return image, label


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


"""
Main function where train and test will be called.
"""
def main() -> None:
    pass



if __name__ == "__main__":
    main()
