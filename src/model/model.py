import os
import pandas as pd
import cv2 as cv
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, parent_dir, transform=None, target_transform=None):
        self.info = self.annotate(parent_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        img_info = self.info.iloc[idx]
        path = os.path.join(img_info["dir"], img_info["file"])

        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        label = img_info["label"]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    @staticmethod
    def annotate(parent_dir):
        dirnames = []
        filenames = []
        labels = []
        for dirname, _, files in os.walk(parent_dir):
            for f in files:
                dirnames.append(dirname)
                filenames.append(f)
                labels.append(os.path.basename(dirname))

        annotation = pd.DataFrame({"dir": dirnames,
                                   "file": filenames,
                                   "label": labels})

        annotation["dir"] = annotation["dir"].astype("category")
        annotation["label"] = annotation["label"].astype("category")

        return annotation
