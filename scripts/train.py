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
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import DataLoader

from caltech_lib.constants import IMAGE_SIZE, DATADIR, bcolors
from caltech_lib.dataloader import Caltech_Dataset, split_dataset, get_dataset_filenames
from caltech_lib.model_architecture import My_Model, VGG_Model
from caltech_lib.train_utils import train_loop, test_loop


def main():
    N_CLASSES = 20
    # learning rate
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    N_EPOCHS = 100
    MODEL_SAVE_FILE = f"/mnt/Terry/ML_models/caltech256_{N_CLASSES}class_classifier.pth"
    # Do transfer learning?
    MODEL_RETRAIN_WEIGHTS = None
    #MODEL_RETRAIN_WEIGHTS = f"/mnt/Terry/ML_models/caltech256_{N_CLASSES}class_classifier.pth"

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # 72.46 vs 5.38 seconds for 5 epochs - without vs with 3090 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop(size=IMAGE_SIZE[:2]),
                                # Augmentations
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomGrayscale(p=0.05),
                                transforms.RandomRotation(degrees=(-100, 100),
                                                          interpolation=InterpolationMode.BILINEAR),
                                # Normalize images the same way
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])
    truth_transform  = transforms.Compose([
                                 transforms.ToTensor(),
                                 #transforms.CenterCrop(size=IMAGE_SIZE[:2]),
                                 transforms.FiveCrop(size=IMAGE_SIZE[:2]),
                                 transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                                 # Normalize images the same way
                                 # https://discuss.pytorch.org/t/trouble-using-transforms-fivecrop-tencrop/32059/4
                                 transforms.Lambda(lambda tensors:
                                    torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])(t)
                                                for t in tensors]))
                                 #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      #std=[0.229, 0.224, 0.225])
                             ])

    filenames = get_dataset_filenames(N_CLASSES, DATADIR)
    # split the filenames into train and test set
    train_files, test_files = split_dataset(filenames, train_test_split=0.8, shuffle=True)

    # Come up with a way to actually split the files
    cd_train = Caltech_Dataset(train_files, DATADIR, image_size=IMAGE_SIZE,
                               transform=image_transform,
                               n_classes=N_CLASSES)

    cd_test = Caltech_Dataset(test_files, DATADIR, image_size=IMAGE_SIZE,
                              transform=truth_transform,
                              n_classes=N_CLASSES)

    train_dataloader = DataLoader(cd_train, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(cd_test, batch_size=BATCH_SIZE // 4, shuffle=False)

    # train a pre-trained model?
    if MODEL_RETRAIN_WEIGHTS is not None:
        model = torch.load(MODEL_RETRAIN_WEIGHTS)
    else:
        model = VGG_Model(N_CLASSES)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Place things on the GPU
    model.to(device)
    loss_function.to(device)

    bestloss = 1e9
    t0 = time.perf_counter()
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}\n-----------")
        train_loop(train_dataloader, model, loss_function, optimizer, device)
        curr_test_loss = test_loop(test_dataloader, model, loss_function, device)

        # Save model if it's better
        if curr_test_loss < bestloss:
            print(f"{bcolors.OKGREEN}{curr_test_loss:.4f}{bcolors.ENDC} > {bestloss:.4f}... \nsaving model to {MODEL_SAVE_FILE}")
            torch.save(model, MODEL_SAVE_FILE)
            bestloss = curr_test_loss

    t1 = time.perf_counter()
    print(f"Time taken for {N_EPOCHS} epochs: {t1 - t0:.5f}")


if __name__ == "__main__":
    main()

