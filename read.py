

import os
import glob
import cv2
import ipdb
import PIL
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import transforms

from model_architecture import My_Model
from constants import DATADIR
from dataloader import read_image


if __name__ == "__main__":
    subdirs = sorted(glob.glob(os.path.join(DATADIR, "**/")))

    model = My_Model(10)
    image_tranform = transforms.Compose([
                                            transforms.ToTensor(),
                                        ])

    # What does the average image look like?
    for subdir in subdirs:
        # glob the curr class images
        filenames = sorted(glob.glob(os.path.join(subdir, "*")))
        classname = os.path.basename(os.path.dirname(subdir)).split(".")[-1]
        print(f"Class: {classname}")

        widths = np.zeros(len(filenames))
        heights = np.zeros(len(filenames))
        for i, filename in enumerate(filenames):
            # read in image and pass to tensor
            print(filename)
            image = read_image(filename)
            #image = image[:256, :256]
            #t_image = torch.unsqueeze(image_tranform(image), dim=0)
            ipdb.set_trace()
            #out = model(t_image)
            h, w, _ = image.shape
            widths[i] = w
            heights[i] = h

        print(f"Heights: ", np.min(heights), np.max(heights), np.mean(heights), np.std(heights))
        print(f"Widths: ", np.min(widths), np.max(widths), np.mean(widths), np.std(widths))



