import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import pandas as pd
import numpy as np


class Net(nn.Module):
    """A custom neural network tailored for Leaffliction project"""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(3, 3)
        self.conv1 = nn.Conv2d(3, 6, 3, stride=2)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2)
        self.fc1 = nn.Linear(12 * 6 * 6, 10)
        self.fc2 = nn.Linear(10, 100)
        self.fc3 = nn.Linear(100, 8)

        self.debug = True

    def forward(self, x):
        if self.debug:  # DEBUG
            print('[Debug] Input', x.shape)  # DEBUG

        x = self.pool(self.conv1(x))
        x = F.relu(x)

        if self.debug:  # DEBUG
            print('[Debug] Conv1', x.shape)  # DEBUG

        x = self.pool(self.conv2(x))
        x = F.relu(x)

        if self.debug:  # DEBUG
            print('[Debug] Conv2', x.shape, end="\n\n")  # DEBUG

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.debug:
            self.debug = False

        return x


def train_net(net: nn.Module,
              trainLoader: DataLoader,
              validationLoader: DataLoader,
              optimizer: torch.optim.Optimizer,
              criterion: nn.modules.loss._Loss,
              device=None,
              epochs=100,
              verbose=True) -> pd.DataFrame:
    """Train a neural network and return performance history"""
    phases = {"train": trainLoader,
              "validation": validationLoader}
    history = {"train_loss": [],
               "val_loss": []}

    for epoch in range(epochs):
        train_loss = 0.
        val_loss = 0.
        val_total = 0
        val_correct = 0

        for phase, loader in phases.items():
            n_samples = 0
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
                if phase == "train":
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    batch_size = inputs.size(0)
                    n_samples += batch_size

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_size

                elif phase == "validation":
                    with torch.no_grad():
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)

                        batch_size = inputs.size(0)
                        n_samples += batch_size

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                # loss_history.append(loss.item())
            if phase == "train":
                train_loss /= n_samples
            elif phase == "validation":
                val_loss /= n_samples
                val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"[epoch {epoch+1:03d}] "
                  f"train_loss: {train_loss:.5f} "
                  f"val_loss: {val_loss:.5f} "
                  f"val_acc: {val_acc:.2%}")

    history |= {"epoch": range(1, epochs + 1)}

    return pd.DataFrame(history).set_index("epoch")


def test_net(net,
             testLoader: DataLoader,
             classes: list[str],
             device=None) -> pd.DataFrame:
    """Return predictions of a model against a test dataset"""
    all_outputs = None
    all_predictions = np.array([])
    all_labels = np.array([])

    recover_training = net.training
    net.eval()

    with torch.no_grad():
        for data in testLoader:
            images, labels = data

            if device:
                images = images.to(device)
                labels = labels.to(device)

            outputs = net(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            _, predictions = torch.max(outputs, 1)

            outputs = outputs.cpu().detach().numpy()

            if all_outputs is None:
                all_outputs = outputs
            else:
                all_outputs = np.concatenate((all_outputs,
                                              outputs))

            all_predictions = np.concatenate((all_predictions,
                                              predictions.cpu()))
            all_labels = np.concatenate((all_labels,
                                         labels.cpu()))

    result = pd.DataFrame(all_outputs,
                          columns=classes[0:all_outputs.shape[1]])

    result["prediction"] = pd.Categorical.from_codes(
        all_predictions.astype(int),
        categories=classes
    )
    result["label"] = pd.Categorical.from_codes(
        all_labels.astype(int),
        categories=classes
    )

    if recover_training:
        net.train()

    return result
