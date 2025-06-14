import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from datetime import timedelta

from dataset import CustomImageDataset, save_classes
from network import Net, train_net, test_net

import os


def _get_device():
    return torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() \
        else "cpu"


def _available_devices():
    devices = {_get_device()}
    devices.add("cpu")

    return devices


def _parse_cmd_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        prog="Model Trainer",
        description="Train Custom Model for Leaffliction Project"
    )

    parser.add_argument("--training-set", required=True)
    parser.add_argument("--validation-set", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-split", type=float, default=.2,
                        help="Splits training set into training-test sets "
                        "if validation-set arg is not provided")
    parser.add_argument("--learning-rate", "--lr", type=float, default=5e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", "--dst", default="output")
    parser.add_argument("--device", default=_get_device(),
                        choices=_available_devices())

    # No args for loss function and optimizer

    args = parser.parse_args()

    if not (args.validation_split > 0 and args.validation_split < 1):
        raise AssertionError("validation-split parameter is not "
                             "within range of (0, 1)")

    return args


def save_history_plot(path, history):
    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(8, 6))

    sns.lineplot(data=history, x="epoch", y="train_loss", label="Training")
    sns.lineplot(data=history, x="epoch", y="val_loss", label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig.tight_layout()
    fig.savefig(path)


def main():

    args = _parse_cmd_arguments()

    device = args.device
    transform = CustomImageDataset.transform_scheme(
        "scheme1"
    )

    data = CustomImageDataset(args.training_set,
                              transform=transform)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    save_classes(os.path.join(args.output_dir, "classes.json"),
                 data.classes)

    if args.validation_set is not None:
        train = data
        test = CustomImageDataset(args.validation_set,
                                  transform=transform)
    else:
        train_test_split = (1. - args.validation_split,
                            args.validation_split)
        gen = torch.Generator().manual_seed(40)
        train, test = torch.utils.data.random_split(data,
                                                    train_test_split,
                                                    generator=gen)

    print("          Dataset Info           ",
          "=================================",
          f"Training   set size : {len(train):10}",
          f"Validation set size : {len(test):10}",
          f"Total               : {len(data):10}",
          sep="\n", end="\n\n")

    print("          Training Info          ",
          "=================================",
          f"Device              : {args.device}",
          f"Learning rate       : {args.learning_rate}",
          f"Batch size          : {args.batch_size}",
          f"Epochs              : {args.epochs}",
          "Optimizer           : Gradient Descent",
          "Loss                : Cross Entropy",
          sep="\n", end="\n\n")

    trainLoader = DataLoader(train, shuffle=True,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True)

    testLoader = DataLoader(test, shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)

    net = Net().to(args.device)
    print(net, end="\n\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate)

    start = time()
    history = train_net(net,
                        trainLoader,
                        testLoader,
                        optimizer,
                        criterion,
                        device=args.device,
                        epochs=args.epochs)
    end = time()

    print()
    print("Training is finished.")
    print("Elapsed time :", timedelta(seconds=(end - start)))

    torch.save(net.state_dict(),
               os.path.join(args.output_dir, "model.pth"))

    result = test_net(net, testLoader, data.classes, device=device)
    accuracy = (result["prediction"] == result["label"]).sum() / len(result)

    print()
    print(f"Accuracy of the network on {len(result)} "
          f"test images: {accuracy:.2%}")

    save_history_plot(os.path.join(args.output_dir, "loss.figure.png"),
                      history)

    print()
    print("Output files are written to", args.output_dir)


if __name__ == "__main__":
    main()
