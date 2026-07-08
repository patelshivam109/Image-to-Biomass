"""
Early Stopping Utility
"""

from pathlib import Path
import torch


class EarlyStopping:

    def __init__(
        self,
        patience=5,
        path=None
    ):

        self.patience = patience

        self.counter = 0

        self.best_loss = float("inf")

        self.early_stop = False

        if path is None:

            project_root = Path(__file__).resolve().parents[2]

            model_dir = project_root / "models"

            model_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            self.path = model_dir / "best_model.pth"

        else:

            self.path = Path(path)

    def __call__(
        self,
        val_loss,
        model
    ):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            self.counter = 0

            torch.save(
                model.state_dict(),
                self.path
            )

        else:

            self.counter += 1

            print(
                f"EarlyStopping Counter: "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.early_stop = True