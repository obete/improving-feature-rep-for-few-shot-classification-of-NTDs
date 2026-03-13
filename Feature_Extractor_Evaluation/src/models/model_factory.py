import torch.nn as nn
from torchvision import models

def create_model(config, num_classes:int):
    """THis function Loads the different pre-trained models.
    In here we freeze the backbonw and attach a head which matches the dataset classes.
    The function returns a dictionary of pre-traine models with an attached linear head"""
    model_list = config.model.list
    model_dict = {}
    for i in model_list:
        i = i.lower()
        if i == "resnet50":
            model_dict["resnet50"] = models.resnet50(weights="IMAGENET1K_V1")
        elif i == "resnet101":
            model_dict["resnet101"] = models.resnet101(weights="IMAGENET1K_V1")
        elif i == "vgg16":
            model_dict["vgg16"]  = models.vgg16(weights="IMAGENET1K_V1")
        elif i == "vgg16bn":
            model_dict[i]  = models.vgg16_bn(weights="IMAGENET1K_V1")
        elif i == "efficientnet_b0":
            model_dict[i]  = models.efficientnet_b0(weights="IMAGENET1K_V1")
        elif i == "efficientnet_b1":
            model_dict[i]  = models.efficientnet_b1(weights="IMAGENET1K_V1")
        elif i == "mobilenet-v2":
            model_dict[i]  = models.mobilenet_v2(weights="IMAGENET1K_V1")
        elif i == "densenet121":
            model_dict[i]  = models.densenet121(weights="IMAGENET1K_V1")
        elif i == "densenet161":
            model_dict[i]  = models.densenet161(weights="IMAGENET1K_V1")
        elif i == "vit_b_16":
            model_dict[i]  = models.vit_b_16(weights="IMAGENET1K_V1")
        elif i == "vit_b_32":
            model_dict[i]  = models.vit_b_32(weights="IMAGENET1K_V1")
        else:
           raise ValueError(f"Unknown Model{i}")
        
    for name, model in model_dict.items():
        for param in model.parameters():
            param.requires_grad = False

        # Adjust classifier layer per architecture
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            layers = list(model.classifier.children())
            if isinstance(layers[-1], nn.Linear):
                in_features = layers[-1].in_features
                layers[-1] = nn.Linear(in_features, num_classes)
                model.classifier = nn.Sequential(*layers)
            else:
                model.classifier = nn.Linear(model.classifier[2].in_features, num_classes)
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif hasattr(model, 'heads'):
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
  

    return model_dict

