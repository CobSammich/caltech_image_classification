"""
Module for training functions and driver for training
"""

import ipdb
import numpy as np
import os
import time

import torch
from torch.nn.modules import loss
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from constants import IMAGE_SIZE, DATADIR
from dataloader import Caltech_Dataset
from model_architecture import My_Model

def train_loop(dataloader: DataLoader,
               model: torch.nn.Module,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    size = len(dataloader.dataset)
    for batch_num, (images, labels) in enumerate(dataloader):
        # make prediction and compute loss on this batch
        pred = model(images.to(device))
        loss = loss_function(pred, labels.float().to(device))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch_num * len(images)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader: DataLoader,
              model: torch.nn.Module,
              loss_function: torch.nn.Module,
              device: torch.device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            pred = model(images.to(device))
            test_loss += loss_function(pred, labels.float().to(device)).item()
            pred = pred.detach().cpu()
            correct += (pred.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    N_CLASSES = 2
    # learning rate
    LEARNING_RATE = 0.001
    N_EPOCHS = 5

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

    # Come up with a way to actually split the files
    cd_train = Caltech_Dataset(DATADIR, image_size=IMAGE_SIZE,
                               transform=image_transform,
                               n_classes=N_CLASSES)
    cd_test = Caltech_Dataset(DATADIR, image_size=IMAGE_SIZE,
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



