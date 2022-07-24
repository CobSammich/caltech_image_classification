"""

"""

from typing import Tuple, List
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

from caltech_lib.constants import DATADIR, IMAGE_SIZE
from caltech_lib.image_utils import read_image, upsample_image, plot_batch, normalize

def get_dataset_filenames(n_classes:int, datadir: str):
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
    class_dirs = sorted(glob.glob(os.path.join(DATADIR, "**/")))[:n_classes]
    filenames = [sorted(glob.glob(os.path.join(d, "*.jpg"))) for d in class_dirs]
    # now flatten the list
    filenames = [imf for imdir in filenames for imf in imdir]
    return filenames


def split_dataset(filenames: List[str],
                  train_test_split: float,
                  shuffle: bool = True) -> Tuple[List[str], List[str]]:
    n_files = len(filenames)
    if shuffle:
        np.random.shuffle(filenames)
    train = filenames[:int(n_files * train_test_split)]
    n_train = len(train)
    test = filenames[n_train:]
    return train, test


class Caltech_Dataset(Dataset):
    def __init__(self,
                 filenames: List[str],
                 datadir: str,
                 image_size: Tuple[int, int] = (256, 256),
                 transform: transforms.Compose = None,
                 n_classes: int = 257) -> None:
        self.filenames = filenames
        self.datadir = datadir
        self.image_size = image_size
        self.transform = transform

        # Retrieve the classnames
        self.class_dirs = sorted(glob.glob(os.path.join(self.datadir, "**/")))[:n_classes]
        # Classnames from "/path/to/002.american-flag/*.jpg"
        self.classnames = [os.path.basename(os.path.dirname(p)).split(".")[-1]
            for p in self.class_dirs]
        # Create one-hot encoding vectors for each class
        self.classname_to_classnum = {k : F.one_hot(torch.tensor(v), num_classes=n_classes)
            for v, k in enumerate(self.classnames)}
        self.classnum_to_classname = {np.argmax(v.numpy()) : k
            for k,v in self.classname_to_classnum.items()}


        if self.transform is None:
            # we still need to convert to a tensor
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        img_path = self.filenames[index]
        # Image label is in the
        dirname = os.path.basename(os.path.dirname(img_path))
        label = dirname.split(".")[-1]
        # Read image and pass through transforms
        image = read_image(img_path)
        image = normalize(image).astype(np.float32)
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

    # Grab all the files in the dataset
    filenames = get_dataset_filenames(n_classes, datadir=DATADIR)
    ipdb.set_trace()
    train_files, test_files = split_dataset(filenames, train_test_split=0.8, shuffle=True)

    tr = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.FiveCrop(size=image_size),
                                transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                                #transforms.RandomCrop(size=image_size)
                                #transforms.Resize(size=min(image_size))
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])

    cd = Caltech_Dataset(filenames=train_files, datadir=DATADIR,
                         image_size=IMAGE_SIZE, transform=tr, n_classes=n_classes)
    ipdb.set_trace()

    train_dataloader = DataLoader(cd, batch_size=32, shuffle=True)
    images, labels = next(iter(train_dataloader))
    ipdb.set_trace()
    label_strings = [cd.classnum_to_classname[np.argmax(label.numpy())] for label in labels]
    plot_batch(images, label_strings)
