#model_factory.py

import torch.nn as nn
from torchvision import models


class ModelFactory:
    """
    This class handles; Loading pretrained torchvision models, freezing the backbone,
    and replaces the classifier layer for a given number of classes as defined by the dataset.
    """

    def __init__(self, config, num_classes: int):

        self.config = config
        self.num_classes = num_classes
        self.model_list = config.model.list

        self.model_dict = {}

    def load_models(self):
        """Loading pretrained models from torchvision."""

        for name in self.model_list:

            name = name.lower()

            if name == "resnet50":
                model = models.resnet50(weights="IMAGENET1K_V1")
            elif name == "resnet101":
                model = models.resnet101(weights="IMAGENET1K_V1")
            elif name == "vgg16":
                model = models.vgg16(weights="IMAGENET1K_V1")
            elif name == "vgg16bn":
                model = models.vgg16_bn(weights="IMAGENET1K_V1")
            elif name == "efficientnet_b0":
                model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            elif name == "efficientnet_b1":
                model = models.efficientnet_b1(weights="IMAGENET1K_V1")
            elif name == "mobilenet_v2":
                model = models.mobilenet_v2(weights="IMAGENET1K_V1")
            elif name == "densenet121":
                model = models.densenet121(weights="IMAGENET1K_V1")
            elif name == "densenet161":
                model = models.densenet161(weights="IMAGENET1K_V1")
            elif name == "vit_b_16":
                model = models.vit_b_16(weights="IMAGENET1K_V1")
            elif name == "vit_b_32":
                model = models.vit_b_32(weights="IMAGENET1K_V1")
            else:
                raise ValueError(f"Unknown model {name}")
            self.model_dict[name] = model

    def freeze_backbone(self):
        """Freezing backbone weights."""

        for model in self.model_dict.values():
            for param in model.parameters():
                param.requires_grad = False

    def replace_classifier(self):
        """Replacing final classifier layer."""

        for model in self.model_dict.values():

            if hasattr(model, "fc"):
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
                layers = list(model.classifier.children())

                if isinstance(layers[-1], nn.Linear):
                    in_features = layers[-1].in_features
                    layers[-1] = nn.Linear(in_features, self.num_classes)

                    model.classifier = nn.Sequential(*layers)
                else:
                    model.classifier = nn.Linear(
                        model.classifier[2].in_features,
                        self.num_classes
                    )

            elif hasattr(model, "classifier"):
                model.classifier = nn.Linear(
                    model.classifier.in_features,
                    self.num_classes
                )
            elif hasattr(model, "heads"):
                model.heads.head = nn.Linear(
                    model.heads.head.in_features,
                    self.num_classes
                )

    def get_model_info(self, model):
        "Here we want to get info about the models we are evaluation, especically the trainable parameters"
        
        if hasattr(model, "fc"):
            out_features = model.fc.out_features

        elif hasattr(model, "classifier"):
            if hasattr(model.classifier, "out_features"):
                out_features = model.classifier.out_features
            else:
                out_features = model.classifier[-1].out_features

        elif hasattr(model, "heads"):
            out_features = model.heads.head.out_features

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return out_features, trainable_params


    def build(self):
        """Full pipeline to create models."""

        self.load_models()
        self.freeze_backbone()
        self.replace_classifier()

        return self.model_dict
    
