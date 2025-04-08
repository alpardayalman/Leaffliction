import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from datetime import timedelta

from dataset import CustomImageDataset, save_encoder, transform_scheme1
from network import Net, train_net, test_net


def main():
    transform = transform_scheme1()

    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() \
        else "cpu"

    data = CustomImageDataset("/home/timur/workspace/test/images",
                              transform=transform)

    save_encoder("label.encoder.npy", data.label_encoder)

    batch_size = 64
    num_workers = 4
    epochs = 50
    train_test_split = [.8, .2]
    learning_rate = 5e-2

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
          "Optimizer           : Gradient Descent",
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
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    start = time()
    history = train_net(net,
                        trainLoader,
                        testLoader,
                        optimizer,
                        criterion,
                        device=device,
                        epochs=epochs)
    end = time()

    print()
    print("Training is finished.")
    print("Elapsed time :", timedelta(seconds=(end - start)))

    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(8, 6))

    sns.lineplot(data=history, x="epoch", y="train_loss", label="Training")
    sns.lineplot(data=history, x="epoch", y="val_loss", label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig.tight_layout()
    fig.savefig("loss.figure.png")

    torch.save(net.state_dict(), "model.pth")

    result = test_net(net, testLoader, device=device)

    print()
    print(f"Accuracy of the network on {result['total']} "
          f"test images: {result['correct'] / result['total']:.2%}")


if __name__ == "__main__":
    main()
