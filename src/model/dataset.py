import os
import pandas as pd
import cv2 as cv
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class CustomImageDataset(Dataset):
    def __init__(self, parent_dir, transform=None, target_transform=None):
        images, labels, label_encoder = self.annotate(parent_dir)
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
    def annotate(parent_dir):
        dirnames = []
        filenames = []
        labels = []
        for dirname, _, files in os.walk(parent_dir):
            for f in files:
                dirnames.append(dirname)
                filenames.append(f)
                labels.append(os.path.basename(dirname))

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        labels = torch.from_numpy(labels)
        
        images = pd.DataFrame({"dir": dirnames,
                               "file": filenames})

        images["dir"] = images["dir"].astype("category")

        return images, labels, label_encoder
