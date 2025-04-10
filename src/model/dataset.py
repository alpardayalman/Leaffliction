import os
import json
import pandas as pd
import cv2 as cv
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    """A Dataset class tailored for loading image files"""
    def __init__(self, parent_dir,
                 transform=None, target_transform=None,
                 classes: list[str] | None = None):
        """Initialize a dataset of images categorized by subdirectories

        Positional Arguments:
        parent_dir       -- path to parent directory of image files

        Keyword Arguments:
        transform        -- function to transform input data
        target_transform -- function to transform label data
        label_encoder    -- predefined label_encoder


        Augmented data can be specified in separate subdirectories,
        for instance:
        "
        images/
        ├── Apple_Black_rot
        ├── Apple_healthy
        ├── Apple_rust
        ├── Apple_scab
        ├── augmentation
        │    ├── Apple_Black_rot
        │    ├── Apple_healthy
        │    ├── Apple_rust
        │    └── Apple_scab
        ...
        "
        """
        images, labels, classes = self.annotate(parent_dir, classes)
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images.iloc[idx]
        path = os.path.join(img["dir"], img["file"])

        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img = torch.from_numpy(img).type(torch.float32)

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    @staticmethod
    def annotate(parent_dir,
                 classes: list[str] | None = None) -> tuple[pd.DataFrame,
                                                            torch.Tensor,
                                                            list[str]]:
        """Return path, label and class information of image files"""
        dirnames = []
        filenames = []
        labels = []
        for dirname, _, files in os.walk(parent_dir):
            for f in files:
                dirnames.append(dirname)
                filenames.append(f)
                labels.append(os.path.basename(dirname))

        if classes:
            new = [lab for lab in set(labels) if lab not in classes]
            if new:
                print("Appending new classes:", new)
                classes.extend(new)

        labels = pd.Categorical(labels, categories=classes)
        classes = labels.categories.tolist()
        labels = torch.from_numpy(labels.codes.astype("int64"))

        images = pd.DataFrame({"dir": dirnames,
                               "file": filenames})

        images["dir"] = images["dir"].astype("category")

        return images, labels, classes

    @staticmethod
    def transform_scheme(scheme) -> transforms.Compose | None:
        """Return a predefined input transformation scheme"""
        return {
            "scheme1": transform_scheme1()
        }.get(scheme)


def save_classes(path, classes: list[str]):
    """Save categories to a file"""
    with open(path, "w") as file:
        json.dump(classes, file)


def load_classes(path) -> list[str]:
    """Load categories from a file"""
    with open(path, "r") as file:
        data = json.load(file)
        return data


def transform_scheme1() -> transforms.Compose:
    """The simplest predefined input transformation scheme"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
