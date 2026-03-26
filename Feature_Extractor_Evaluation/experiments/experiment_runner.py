#experiment_runner.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Subset
from data.datasets import DatasetFactory
from models.model_factory import ModelFactory
from engine.trainer import Trainer 



class ExperimentRunner(DatasetFactory, ModelFactory, Trainer):
 
    def __init__(self, data_config, model_config, batch_size=16, epochs=50, lr=1e-3, patience=5, min_delta=1e-3, seed=42, device=None):

        self.data_config = data_config
        self.model_config = model_config
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        self.seed = seed
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        # -------------------------
        # Build dataset
        # -------------------------
        dataset_builder = DatasetFactory(self.data_config)
        train_ds, val_ds, num_classes, class_names = dataset_builder.build()

        print("\nDataset Loaded")
        print("Classes:", num_classes)
        print(class_names)

        #loading the Model  
        all_models = ModelFactory(self.model_config, num_classes)
        models_dict = all_models.build()


        #Set Dataloader
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

        #Training
        summary_rows = []

        for name, model in models_dict.items():

            print(f"\nTraining {name}")

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)

            trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, self.device)

            history = trainer.fit(self.epochs)

            best_epoch = max(history, key=lambda x: x["weighted_f1"])

            summary_rows.append({
                "Model": name,
                "Best_Accuracy": best_epoch["accuracy"],
                "Best_Macro_F1": best_epoch["macro_f1"],
                "Best_Weighted_F1": best_epoch["weighted_f1"],
            })

        summary_df = pd.DataFrame(summary_rows)

        summary_df = summary_df.sort_values(
            "Best_Weighted_F1", ascending=False
        )

        return summary_df