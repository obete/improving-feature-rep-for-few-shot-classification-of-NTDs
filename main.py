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
from models.model_factory import ModelFactory
from engine.trainer import Trainer   # adjust if trainer.py location differs


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

    # -------------------------
    # Build dataset
    # -------------------------
    dataset_builder = DatasetFactory(data_config)

    train_ds, val_ds, num_classes, class_names = dataset_builder.build()

    print("\nDataset Loaded")
    print("Classes:", num_classes)
    print(class_names)

    # -------------------------
    # Dataloaders
    # -------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=data_config.data.batch_size,
        shuffle=True,
        num_workers=data_config.data.num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_config.data.batch_size,
        shuffle=False,
        num_workers=data_config.data.num_workers
    )

    # -------------------------
    # Build models
    # -------------------------
    factory = ModelFactory(model_config, num_classes=num_classes)

    models_dict = factory.build()

    results = []

    # -------------------------
    # Train each model
    # -------------------------
    for name, model in models_dict.items():

        print(f"\nTraining {name}")

        out_features, params = factory.get_model_info(model)

        print(
            f"{name} | Trainable Params: {params:,} | Output Features: {out_features}"
        )

        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3
        )

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            DEVICE
        )

        history = trainer.fit(epochs=10)

        final_metrics = trainer.evaluate()

        results.append({
            "model": name,
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "weighted_f1": final_metrics["weighted_f1"]
        })

        print(
            f"{name} Final → "
            f"Acc: {final_metrics['accuracy']:.4f} | "
            f"Macro F1: {final_metrics['macro_f1']:.4f}"
        )

    # -------------------------
    # Summary
    # -------------------------
    print("\nModel Comparison\n")

    for r in results:
        print(
            f"{r['model']:15} | "
            f"Acc: {r['accuracy']:.4f} | "
            f"Macro F1: {r['macro_f1']:.4f} | "
            f"Weighted F1: {r['weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()