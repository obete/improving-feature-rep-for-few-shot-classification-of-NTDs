import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, "Feature_Extractor_Evaluation/src"))
sys.path.append(BASE_DIR)

from omegaconf import OmegaConf

from data.datasets import DatasetFactory
from Feature_Extractor_Evaluation.experiments import experiment_runner


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    # -------------------------
    # Load configs
    # -------------------------
    model_config = OmegaConf.load(
        os.path.join(BASE_DIR, "Feature_Extractor_Evaluation/configs/models_list.yaml")
    )

    data_config = OmegaConf.load(
        os.path.join(BASE_DIR, "Feature_Extractor_Evaluation/configs/fitzpatrick17k.yaml")
    )

    experiment_summary = experiment_runner.ExperimentRunner(data_config, model_config)
    print(experiment_summary)



if __name__ == "__main__":
    main()