import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1)

        return t


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


@torch.no_grad()
def get_all_preds(model, loader):
    # Function that provides predictions for the entire dataset
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds


# Adjust output width
numpy.set_printoptions(linewidth=120)
torch.set_printoptions(linewidth=120)

# Prepare dataset
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    , train=True
    , download=True
    , transform=transforms.Compose(
        [transforms.ToTensor()]
    )
)

# Turn on gradient calculation
torch.set_grad_enabled(True)

# Prepare instance of network
network = Network()

# Prepare data loader
data_loader = torch.utils.data.DataLoader(
    train_set, batch_size=100
)

# Prepare optimizer
optimizer = optim.Adam(network.parameters(), lr=0.01)

print("Training Network...")
for epoch in range(3):

    total_loss = 0
    total_correct = 0

    for batch in data_loader:
        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds, labels)  # Calculating the loss

        # Clear out accumulated gradients
        optimizer.zero_grad()

        # Calculate the gradients
        loss.backward()

        # Update Weights
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch:", epoch, "total correct:", total_correct, "total loss", total_loss,
          "Accuracy", total_correct/len(train_set))


train_preds = get_all_preds(network, data_loader)
preds_correct = get_num_correct(train_preds, train_set.targets)
print('total correct:', preds_correct)
print("accuracy:", preds_correct / len(train_set.targets))

print()
print("Building confusion matrix...")
stacked = torch.stack(
    (train_set.targets, train_preds.argmax(dim=1))
    , dim=1
)
cmt = torch.zeros(10, 10, dtype=torch.int64)
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
print(cmt)




