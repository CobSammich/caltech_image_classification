"""
Driver script for training
"""

import ipdb
import numpy as np
import os
import sys
import time

import torch
from torch.nn.modules import loss
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from caltech_lib.constants import IMAGE_SIZE, DATADIR
from caltech_lib.dataloader import Caltech_Dataset, split_dataset, get_dataset_filenames
from caltech_lib.model_architecture import My_Model
from caltech_lib.train_utils import train_loop, test_loop


def main():
    N_CLASSES = 20
    # learning rate
    LEARNING_RATE = 0.0001
    N_EPOCHS = 100

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # 72.46 vs 5.38 seconds - without vs with 3090 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop(size=IMAGE_SIZE[:2])
                                #transforms.Resize(size=min(image_size))
                            ])
    truth_transform  = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.CenterCrop(size=IMAGE_SIZE[:2])
                             ])

    filenames = get_dataset_filenames(N_CLASSES, DATADIR)
    # split the filenames into train and test set
    train_files, test_files = split_dataset(filenames, train_test_split=0.8, shuffle=True)
    ipdb.set_trace()

    # Come up with a way to actually split the files
    cd_train = Caltech_Dataset(train_files, DATADIR, image_size=IMAGE_SIZE,
                               transform=image_transform,
                               n_classes=N_CLASSES)

    cd_test = Caltech_Dataset(test_files, DATADIR, image_size=IMAGE_SIZE,
                              transform=truth_transform,
                              n_classes=N_CLASSES)

    train_dataloader = DataLoader(cd_train, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(cd_test, batch_size=32, shuffle=False)

    model = My_Model(N_CLASSES)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Place things on the GPU
    model.to(device)
    loss_function.to(device)

    t0 = time.perf_counter()
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}\n-----------")
        train_loop(train_dataloader, model, loss_function, optimizer, device)
        test_loop(test_dataloader, model, loss_function, device)
    t1 = time.perf_counter()
    print(f"Time taken for {N_EPOCHS} epochs: {t1 - t0:.5f}")


if __name__ == "__main__":
    main()

