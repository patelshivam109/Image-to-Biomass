

from torch.utils.data import DataLoader

from src.config import (
    TRAIN_CSV,
    VALIDATION_CSV,
    BATCH_SIZE,
    NUM_WORKERS
)

from src.data.dataset import BiomassDataset
from src.data.transforms import (
    train_transform,
    val_transform
)


def get_dataloaders():

    train_dataset = BiomassDataset(
        csv_file=TRAIN_CSV,
        transform=train_transform
    )

    val_dataset = BiomassDataset(
        csv_file=VALIDATION_CSV,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader