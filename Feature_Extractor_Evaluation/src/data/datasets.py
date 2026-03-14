#datasets.py

from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import numpy as np


class DatasetFactory:
    """A Class handling all data operations for the Project. Loading, light augmentation and trasnformation, 
       data split into Train and validation set, Determining the number of dataclasses and the class names"""

    def __init__(self, config):
        self.config = config
        self.name = config.data.name.lower()
        self.root = config.data.root
        self.val_size = getattr(config.data, "val_size", 0.2)
        self.seed = getattr(config.data, "seed", 42)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.dataset = None
        self.train_ds = None
        self.val_ds = None

    def load_dataset(self):

        self.dataset = ImageFolder(
            root=self.root,
            transform=self.transform
        )

        return self.dataset

    def make_split(self):

        labels = np.array([y for _, y in self.dataset.samples], dtype=np.int64)

        idx = np.arange(len(self.dataset))

        train_idx, val_idx = train_test_split(
            idx,
            test_size=self.val_size,
            stratify=labels,
            random_state=self.seed
        )

        self.train_ds = Subset(self.dataset, train_idx)
        self.val_ds = Subset(self.dataset, val_idx)

        return self.train_ds, self.val_ds

    def get_metadata(self):

        num_classes = len(self.dataset.classes)
        class_names = self.dataset.classes

        return num_classes, class_names

    def build(self):

        self.load_dataset()
        train_ds, val_ds = self.make_split()
        num_classes, class_names = self.get_metadata()

        return train_ds, val_ds, num_classes, class_names


