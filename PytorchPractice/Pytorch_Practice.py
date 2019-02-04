import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    , train=True
    , download=True
    , transform=transforms.Compose(
        [transforms.ToTensor()]
    )
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
)

torch.set_printoptions(linewidth=120)

# print(len(train_set))
# print(train_set.train_labels)

sample = next(iter(train_set))
image, label = sample
# print(len(sample))
# print(type(sample))
# print(image.shape)
# plt.imshow(image.squeeze(), cmap='gray')
# plt.show()

batch = next(iter(train_loader))
images, labels = batch
print(images.shape)
print(labels.shape)
grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()
