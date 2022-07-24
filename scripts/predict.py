"""
Driver for making predictions using trained caltech256 models
"""

import os
import torch
from torchvision import transforms
import numpy as np
import ipdb
import matplotlib.pyplot as plt

from caltech_lib.image_utils import read_image, normalize, upsample_image
from caltech_lib.constants import DATADIR, IMAGE_SIZE, CLASSNUM_TO_CLASSNAME

# Params
MODEL_SAVE_FILE = f"/mnt/Terry/ML_models/caltech256_20class_classifier_hold.pth"
IMAGE_FILE = "/mnt/Terry/data/caltech_256/256_ObjectCategories/001.ak47/001_0027.jpg"
#

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    model = torch.load(MODEL_SAVE_FILE)
    model.to(device)

    image = read_image(IMAGE_FILE)
    if image.shape[0] < IMAGE_SIZE[0] or image.shape[1] < IMAGE_SIZE[1]:
        image = upsample_image(image, IMAGE_SIZE)

    plt.imshow(image)
    plt.show()

    # make into tensor
    truth_transform  = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.CenterCrop(size=IMAGE_SIZE[:2]),
                                 # Normalize images the same way
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ])

    image = truth_transform(image)
    image = torch.unsqueeze(image, dim=0)

    model.eval()
    pred = model(image.float().to(device))
    pred = pred.detach().cpu().numpy()

    print(f"predicted {CLASSNUM_TO_CLASSNAME[np.argmax(pred) + 1]} with {np.max(pred)*100:.2f} confidence")


if __name__ == "__main__":
    main()
