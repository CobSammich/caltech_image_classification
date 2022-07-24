"""
Module for training functions and driver for training
"""

import ipdb
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np

from caltech_lib.image_utils import plot_batch
from caltech_lib.constants import IMAGE_SIZE

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

        #label_strings = [dataloader.dataset.classnum_to_classname[np.argmax(label.numpy())] for label in labels]
        #plot_batch(images, label_strings)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch_num * len(images)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end='\r')
    print("\n")

def test_loop(dataloader: DataLoader,
              model: torch.nn.Module,
              loss_function: torch.nn.Module,
              device: torch.device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            curr_batch_size = images.shape[0]
            n_crops = 1
            if len(images.shape) == 5:
                n_crops = images.shape[1]
                # Using some type of FiveCrop/TenCrop transform - make
                images = images.view(-1, IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1])
            pred = model(images.to(device))

            # https://discuss.pytorch.org/t/confused-on-how-to-keep-labels-paired-after-using-five-or-tencrop-augmentation/21289/2
            pred_avg = pred.view(curr_batch_size, n_crops, -1).mean(1) # avg over crops

            test_loss += loss_function(pred_avg, labels.float().to(device)).item()
            pred_avg = pred_avg.detach().cpu()
            correct += (pred_avg.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss
