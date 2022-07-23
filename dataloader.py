"""

"""

from typing import Tuple
import os
import glob
import ipdb
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from constants import DATADIR, IMAGE_SIZE
from image_utils import read_image, upsample_image, plot_batch



class Caltech_Dataset(Dataset):
    """
    The dataset is in the format:
    caltech/
        - class1
            - image001
            - image002
            - ...
            - image???
        - class2
        ...
        - class?
    """
    def __init__(self,
                 datadir: str,
                 image_size: Tuple[int, int] = (256, 256),
                 transform: transforms.Compose = None,
                 n_classes: int = 257) -> None:
        self.datadir = datadir
        self.image_size = image_size
        self.transform = transform

        # Retrieve all the filenames of the images
        self.class_dirs = sorted(glob.glob(os.path.join(self.datadir, "**/")))[:n_classes]
        # Classnames from "/path/to/002.american-flag/*.jpg"
        self.classnames = [os.path.basename(os.path.dirname(p)).split(".")[-1]
            for p in self.class_dirs]
        # Create one-hot encoding vectors for each class
        self.classname_to_classnum = {k : F.one_hot(torch.tensor(v), num_classes=n_classes)
            for v, k in enumerate(self.classnames)}

        self.all_filenames = [sorted(glob.glob(os.path.join(d, "*.jpg"))) for d in self.class_dirs]
        self.all_filenames = [imf for imdir in self.all_filenames for imf in imdir]
        if self.transform is None:
            # we still need to convert to a tensor
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        img_path = self.all_filenames[index]
        # Image label is in the
        dirname = os.path.basename(os.path.dirname(img_path))
        label = dirname.split(".")[-1]
        # Read image and pass through transforms
        image = read_image(img_path)
        # check if this image is the right size
        if image.shape[0] < self.image_size[0] or image.shape[1] < self.image_size[1]:
            image = upsample_image(image, self.image_size)

        # If image is grayscale, copy single channel to all channels
        image = self.transform(image)
        label = self.classname_to_classnum[label]
        return image, label



if __name__ == "__main__":
    image_size = IMAGE_SIZE[:2]
    n_classes = 2

    tr = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop(size=image_size)
                                #transforms.Resize(size=min(image_size))
                            ])

    cd = Caltech_Dataset(DATADIR, image_size=IMAGE_SIZE, transform=tr, n_classes=n_classes)
    ipdb.set_trace()

    train_dataloader = DataLoader(cd, batch_size=32, shuffle=True)
    images, labels = next(iter(train_dataloader))
    plot_batch(images, labels)
    ipdb.set_trace()


