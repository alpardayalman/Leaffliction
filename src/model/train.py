import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from dataset import CustomImageDataset

from time import time
from datetime import timedelta

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 12, 3, stride=2)
        self.conv2 = nn.Conv2d(12, 10, 3, stride=2)
        self.conv3 = nn.Conv2d(10, 10, 3, stride=2)
        self.conv4 = nn.Conv2d(10, 10, 3, stride=2)
        self.fc1 = nn.Linear(10 * 15 * 15, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 8)

        self.debug = True

    def forward(self, x):
        if self.debug:  # DEBUG
            print('[Debug] Input', x.shape)  # DEBUG

        x = F.relu(self.conv1(x))

        if self.debug:  # DEBUG
            print('[Debug] Conv1', x.shape)  # DEBUG

        x = F.relu(self.conv2(x))

        if self.debug:  # DEBUG
            print('[Debug] Conv2', x.shape)  # DEBUG

        x = F.relu(self.conv3(x))

        if self.debug:  # DEBUG
            print('[Debug] Conv3', x.shape)  # DEBUG

        x = F.relu(self.conv4(x))

        if self.debug:  # DEBUG
            print('[Debug] Conv4', x.shape, end="\n\n")  # DEBUG

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        if self.debug:
            self.debug = False

        return x


def train_net(net: nn.Module,
              trainLoader: DataLoader,
              validationLoader: DataLoader,
              optimizer: torch.optim.Optimizer,
              criterion: nn.modules.loss._Loss,
              device=None,
              epochs=100):
    phases = {"train": trainLoader,
              "validation": validationLoader}
    history = {"train_loss": [],
               "val_loss": []}

    start = time()

    for epoch in range(epochs):
        train_loss = 0.
        val_loss = 0.
        val_total = 0
        val_correct = 0

        for phase, loader in phases.items():
            for data in loader:
                inputs, labels = data

                if device:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                if phase == "train":
                    net.train()
                    optimizer.zero_grad()
                elif phase == "validation":
                    net.eval()
                else:
                    raise NotImplementedError(f"unimplemented phase {phase}")

                # forward + backward + optimization
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                elif phase == "validation":
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                # loss_history.append(loss.item())

        train_loss /= len(trainLoader.dataset)
        val_loss /= len(validationLoader.dataset)
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[epoch {epoch+1:03d}] "
              f"train_loss: {train_loss:.5f} "
              f"val_loss: {val_loss:.5f} "
              f"val_acc: {val_acc:.2%}")

    end = time()

    print("Training is finished.")
    print("Elapsed time :", timedelta(seconds=(end - start)))

    history |= {"epoch": range(1, epochs + 1)}

    return pd.DataFrame(history).set_index("epoch")


def test_net(net, testLoader, device=None):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testLoader:
            images, labels = data

            if device:
                images = images.to(device)
                labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on {total} "
          f"test images: {100 * correct // total}%")


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() \
        else "cpu"

    data = CustomImageDataset("/home/timur/workspace/test/images",
                              transform=transform)

    batch_size = 64
    num_workers = 4
    epochs = 20
    train_test_split = [.8, .2]
    learning_rate = 0.001

    gen = torch.Generator().manual_seed(40)
    train, test = torch.utils.data.random_split(data,
                                                train_test_split,
                                                generator=gen)

    print("          Dataset Info           ",
          "=================================",
          f"Training   set size : {len(train):10} ({train_test_split[0]:.0%})",
          f"Validation set size : {len(test):10} ({train_test_split[1]:.0%})",
          f"Total               : {len(data):10}",
          sep="\n", end="\n\n")

    print("          Training Info          ",
          "=================================",
          f"Device              : {device}",
          f"Learning rate       : {learning_rate}",
          f"Batch size          : {batch_size}",
          f"Epochs              : {epochs}",
          "Optimizer           : Adam",
          "Loss                : Cross Entropy",
          sep="\n", end="\n\n")

    trainLoader = DataLoader(train, shuffle=True,
                             batch_size=batch_size,
                             num_workers=num_workers)

    testLoader = DataLoader(test, shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers)

    net = Net().to(device)
    print(net, end="\n\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    history = train_net(net,
                        trainLoader,
                        testLoader,
                        optimizer,
                        criterion,
                        device=device,
                        epochs=epochs)

    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(8, 6))

    sns.lineplot(data=history, x="epoch", y="train_loss", label="Training")
    sns.lineplot(data=history, x="epoch", y="val_loss", label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig.tight_layout()
    fig.savefig("loss.figure.png")

    torch.save(net.state_dict(), "mytestmodel.pth")

    test_net(net, testLoader, device=device)


if __name__ == "__main__":
    main()
