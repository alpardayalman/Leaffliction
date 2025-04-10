import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from dataset import CustomImageDataset, load_encoder

from network import Net, test_net


def _get_device():
    return torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() \
        else "cpu"


def _available_devices():
    devices = {_get_device()}
    devices.add("cpu")

    return devices


def benchmark(loader, verbose=False):
    pass


def _parse_cmd_arguments(arguments=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(
        prog="Image Classifier",
        description="Predict category of an image by a model"
    )

    parser.add_argument("--model-path", "-m", required=True)
    parser.add_argument("--input-path", "-i", required=True)
    parser.add_argument("--encoder-path", "-e", default=None)
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="benchmark model against a dataset")
    parser.add_argument("--device", default=_get_device(),
                        choices=_available_devices())
    parser.add_argument("--transformation-scheme", default="scheme1",
                        choices=["scheme1"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args(arguments)

    if args.benchmark and not os.path.isdir(args.input_path):
        raise AssertionError("input path is expected to be a directory "
                             "when benchmark (-b) option is enabled")

    return args


def _test_from_args(args):
    net = Net().to(args.device)
    net.load_state_dict(torch.load(args.model_path, weights_only=True))

    if args.encoder_path:
        encoder = load_encoder(args.encoder_path)
    else:
        encoder = None
        print("Encoder is not provided, model predictions might mismatch.")

    transform = CustomImageDataset.transform_scheme(
        args.transformation_scheme
    )
    test = CustomImageDataset(args.input_path,
                              transform=transform,
                              label_encoder=encoder)
    testLoader = DataLoader(test, shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    return test_net(net,
                    testLoader,
                    test.label_encoder,
                    device=args.device)


def _result_accuracy(result: pd.DataFrame) -> pd.DataFrame:
    among_all = {
        "Overall": {
            "correct": (result["prediction"] == result["label"]).sum(),
            "total": len(result)
        }
    }
    among_classes = {}

    classes = result["label"].cat.remove_unused_categories()
    classes = classes.cat.categories

    for c in classes:
        subset = result.loc[result["label"] == c]
        among_classes[c] = {
            "correct": (subset["prediction"] == subset["label"]).sum(),
            "total": len(subset)
        }

    among_all = pd.DataFrame(among_all).T
    among_classes = pd.DataFrame(among_classes).T

    merged = pd.concat((among_classes, among_all))
    merged["accuracy"] = merged["correct"] / merged["total"]

    return merged


def main():
    args = _parse_cmd_arguments()
    result = _test_from_args(args)

    if args.benchmark:
        result_accuracy = _result_accuracy(result)

        output = result_accuracy.to_string(formatters={
            "accuracy": "{:,.2%}".format
        })
        print(output)
    else:
        print(result)


if __name__ == "__main__":
    main()
