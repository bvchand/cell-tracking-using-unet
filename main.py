
from globals import *
from data_extraction_torch import *
from unet_model import *
from model_train import *



"""
Reference:
https://www.kaggle.com/paultimothymooney/identification-and-segmentation-of-nuclei-in-cells
https://www.kaggle.com/vbookshelf/simple-cell-segmentation-with-keras-and-u-net
"""

if __name__ == '__main__':
    train_dataloader, test_dataloader = data_extraction_torch()
    model = UNet(1, 1).to(device)
    best_model = train_model(model, train_dataloader)


