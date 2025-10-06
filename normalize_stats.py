import torch
from torch.utils.data import DataLoader
import numpy as np
import config
from dataloaders import ArcticLabeledImagePatchDataset

def compute_mean_std(dataset):
    print("Computing mean and std...")

    mean = 0.0
    std = 0.0
    nb_samples = 0

    for i, (images, _) in enumerate(DataLoader(dataset, batch_size=16, shuffle=False)):
        # images: [B, C, H, W]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # [B, C, H*W]

        mean += images.mean(2).sum(0)  # sum over batch
        std += images.std(2).sum(0)
        nb_samples += batch_samples

        if i % 100 == 0:
            print(f"Processed batch {i}")

    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(), std.numpy()


if __name__ == "__main__":
    train_dataset = ArcticLabeledImagePatchDataset(
        config.IMAGE_GRIDS_PATH,
        config.LABEL_GRIDS_PATH,
        config.train_grids,
        config.patch_size,
        config.stride,
        transform=None,
    	normalize=False  # Raw values used to compute mean/std
    )

    mean, std = compute_mean_std(train_dataset)

    np.save(f"experiments/{config.experiment_id}/train_band_mean.npy", mean)
    np.save(f"experiments/{config.experiment_id}/train_band_std.npy", std)

    print("Mean per band:", mean)
    print("Std per band:", std)
