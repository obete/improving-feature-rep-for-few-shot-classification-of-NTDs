#experiment_runner.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


class ExperimentRunner:
 
    def __init__(
        self,
        dataset,
        model_factory,
        batch_size=16,
        epochs=50,
        lr=1e-3,
        patience=5,
        min_delta=1e-3,
        seed=42,
        device=None
    ):

        self.dataset = dataset
        self.model_factory = model_factory

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        self.seed = seed

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_split(self, val_size=0.2):

        labels = [y for _, y in self.dataset.samples]

        train_idx, val_idx = train_test_split(
            range(len(labels)),
            test_size=val_size,
            stratify=labels,
            random_state=self.seed
        )

        return train_idx, val_idx

    def run(self):

        summary_rows = []

        models_dict = self.model_factory.build()

        train_idx, val_idx = self.make_split()

        train_loader = DataLoader(
            Subset(self.dataset, train_idx),
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            Subset(self.dataset, val_idx),
            batch_size=self.batch_size,
            shuffle=False
        )

        for name, model in models_dict.items():

            print(f"\nTraining {name}")

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.lr
            )

            trainer = Trainer(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                self.device
            )

            history = trainer.fit(self.epochs)

            best_epoch = max(history, key=lambda x: x["accuracy"])

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