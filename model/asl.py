import cv2

import numpy as np
import pandas as pd

transform = transforms.Compose([
    transforms.Resize(512, 512) # All images in the dataset are 512x512
])