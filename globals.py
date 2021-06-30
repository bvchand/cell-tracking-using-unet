import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import cv2
import numpy as np
import sys
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")


# Whether to use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image dimensions
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# folders
root = "/Users/bharathi/Desktop/CMPE/Summer2021/summer-project-Gao/"
dataset_path = root + "cell_tracking/"
train_data = dataset_path + "train/"
train700_data = dataset_path + "train_600/"
test_data = dataset_path + "test/"
test700_data = dataset_path + "test_600/"

# model params
BATCH = 1
CRITERION = nn.BCEWithLogitsLoss()
EPOCHS = 10
MODEL_PATH = './model/unet_model.pt'