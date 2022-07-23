"""
Module for training functions and driver for training
"""

import torch
from torch.utils.data.dataloader import DataLoader

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
