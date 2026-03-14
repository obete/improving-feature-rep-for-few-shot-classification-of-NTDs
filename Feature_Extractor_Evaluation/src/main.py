#main.py

from omegaconf import OmegaConf
from data.datasets import DatasetFactory
from models.model_factory import ModelFactory

if __name__ == "__main__":


    config = OmegaConf.load("D:/MSCS_Research/CODE/Feature_Extractor_Evaluation/configs/models_list.yaml")
    config1 = OmegaConf.load("D:/MSCS_Research/CODE/Feature_Extractor_Evaluation/configs/fitzpatrick17k.yaml")
    dataset_builder = DatasetFactory(config1)
    train_ds, val_ds, num_classes, class_names = dataset_builder.build()

    factory = ModelFactory(config, num_classes=num_classes)

    models_dict = factory.build()
    

    for name, model in models_dict.items():
        out_features, params = factory.get_model_info(model) 
        print(name, " | Trainable Parameters: ", params, " | Output features: ", out_features )
