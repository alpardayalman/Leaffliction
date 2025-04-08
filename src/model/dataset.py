import os
import pandas as pd
import cv2 as cv
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from numpy import typing as npt
import numpy as np

import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    """A Dataset class tailored for loading image files"""
    def __init__(self, parent_dir,
                 transform=None, target_transform=None,
                 label_encoder: LabelEncoder | None = None):
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
        images, labels, label_encoder = self.annotate(parent_dir,
                                                      label_encoder)
        self.images = images
        self.labels = labels
        self.label_encoder = label_encoder
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
    def annotate(parent_dir, label_encoder) -> tuple[pd.DataFrame,
                                                     torch.Tensor,
                                                     LabelEncoder]:
        """Return path, label and label encoder information of image files"""
        dirnames = []
        filenames = []
        labels = []
        for dirname, _, files in os.walk(parent_dir):
            for f in files:
                dirnames.append(dirname)
                filenames.append(f)
                labels.append(os.path.basename(dirname))

        if label_encoder is None:
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)
        else:
            labels = label_encoder.transform(labels)

        labels = torch.from_numpy(labels)

        images = pd.DataFrame({"dir": dirnames,
                               "file": filenames})

        images["dir"] = images["dir"].astype("category")

        return images, labels, label_encoder


def save_encoder(path, encoder: LabelEncoder | npt.ArrayLike):
    """Save label encoder to a file"""
    if isinstance(encoder, LabelEncoder):
        if hasattr(encoder, "classes_") \
           and isinstance(encoder.classes_, np.ndarray):
            encoder = encoder.classes_
        else:
            raise AssertionError("encoder is not fit")

    np.save(path, encoder)


def load_encoder(path) -> LabelEncoder:
    """Load label encoder from a file"""
    array = np.load(path)
    encoder = LabelEncoder()
    encoder.fit(np.array(array))

    return encoder


def transform_scheme1() -> transforms.Compose:
    """The simplest predefined input transformation scheme"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
