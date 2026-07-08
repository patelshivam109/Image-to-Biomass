"""
Early Stopping Utility
"""

import torch


class EarlyStopping:

    def __init__(
        self,
        patience=5,
        path="models/best_model.pth"
    ):

        self.patience = patience

        self.path = path

        self.counter = 0

        self.best_loss = float("inf")

        self.early_stop = False

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