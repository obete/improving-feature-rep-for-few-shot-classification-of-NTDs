import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Trainer:

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer


    def train_one_epoch(self):

        self.model.train()

        total_loss = 0
        preds = []
        trues = []
        n = 0

        for x, y in self.train_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            out = self.model(x)
            loss = self.criterion(out, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

            preds.append(out.argmax(1).detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        acc = accuracy_score(trues, preds)

        return {
            "loss": total_loss / max(n, 1),
            "accuracy": acc
        }


    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        total_loss = 0
        preds = []
        trues = []
        n = 0

        for x, y in self.val_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

            preds.append(out.argmax(1).cpu().numpy())
            trues.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        acc = accuracy_score(trues, preds)

        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            trues, preds, average="macro", zero_division=0
        )

        p_w, r_w, f_w, _ = precision_recall_fscore_support(
            trues, preds, average="weighted", zero_division=0
        )

        return {
            "loss": total_loss / max(n, 1),
            "accuracy": acc,
            "macro_precision": p_macro,
            "macro_recall": r_macro,
            "macro_f1": f_macro,
            "weighted_precision": p_w,
            "weighted_recall": r_w,
            "weighted_f1": f_w
        }


    def fit(self, epochs):

        history = []

        for epoch in range(epochs):

            train_metrics = self.train_one_epoch()
            val_metrics = self.evaluate()

            metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics
            }

            history.append(metrics)

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f}"
            )

        return history