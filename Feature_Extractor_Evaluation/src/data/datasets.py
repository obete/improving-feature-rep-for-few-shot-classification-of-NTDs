from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data  import Subset

import numpy as np
from omegaconf import OmegaConf

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def make_split(dataset, val_size=0.2, seed=42):
    # stratified split using dataset labels
    labels = np.array([y for _, y in dataset.samples], dtype=np.int64)
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, stratify=labels, random_state=seed
    )
    return train_idx.tolist(), val_idx.tolist()


transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ])

def create_datasets(config):
    name = config.data.name.lower()
    dataset = ImageFolder(config.data.root, transform=None)
    train_idx, val_idx = train_test_split(dataset)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    num_classes = len(dataset.classes)
    class_names = getattr(dataset, 'classes', None)
    

    return train_ds,val_ds, num_classes, class_names



    
