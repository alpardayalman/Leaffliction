import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from datetime import timedelta

from dataset import CustomImageDataset, save_encoder, transform_scheme1
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

    parser.add_argument("image_dir")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-split", type=float, default=.2)
    parser.add_argument("--learning-rate", "--lr", type=float, default=5e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", "--dst", default="output")
    parser.add_argument("--device", default=_get_device(),
                        choices=_available_devices())
    parser.add_argument("--transformation-scheme", default="scheme1",
                        choices=["scheme1"])

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
    transform = {
        "scheme1": transform_scheme1()
    }.get(args.transformation_scheme)

    data = CustomImageDataset(args.image_dir,
                              transform=transform)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    save_encoder(os.path.join(args.output_dir, "label.encoder.npy"),
                 data.label_encoder)

    train_test_split = (1. - args.validation_split,
                        args.validation_split)

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
          f"Device              : {args.device}",
          f"Learning rate       : {args.learning_rate}",
          f"Batch size          : {args.batch_size}",
          f"Epochs              : {args.epochs}",
          "Optimizer           : Gradient Descent",
          "Loss                : Cross Entropy",
          sep="\n", end="\n\n")

    trainLoader = DataLoader(train, shuffle=True,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    testLoader = DataLoader(test, shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

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

    result = test_net(net, testLoader, device=device)

    print()
    print(f"Accuracy of the network on {result['total']} "
          f"test images: {result['correct'] / result['total']:.2%}")

    save_history_plot(os.path.join(args.output_dir, "loss.figure.png"),
                      history)

    print()
    print("Output files are written to", args.output_dir)


if __name__ == "__main__":
    main()
