import os
import torch


class CheckpointManager:

    def __init__(self, checkpoint_dir, device):
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, best_acc, model_name):

        path = os.path.join(self.checkpoint_dir, f"{model_name}_checkpoint.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc
        }, path)

    def load(self, model, optimizer, model_name):

        path = os.path.join(self.checkpoint_dir, f"{model_name}_checkpoint.pt")

        if os.path.exists(path):

            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print(
                f"Resumed {model_name} from epoch "
                f"{checkpoint['epoch']} (best acc: {checkpoint['best_acc']:.3f})"
            )

            return checkpoint["epoch"], checkpoint["best_acc"]

        return 0, 0