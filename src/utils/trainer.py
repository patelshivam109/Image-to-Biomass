"""
Training Utilities
"""

import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for batch in tqdm(dataloader):

        images = batch["image"].to(device)

        metadata = batch["metadata"].to(device)

        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(images, metadata)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)

            metadata = batch["metadata"].to(device)

            targets = batch["targets"].to(device)

            outputs = model(images, metadata)

            loss = criterion(outputs, targets)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss