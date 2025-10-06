# efficient_training.py

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp      # note the lowercase alias!
from dataloaders import ArcticLabeledImagePatchDataset
import config

device = 'cuda'

# 1) Instantiate SMP‑Unet with EfficientNet‑B4 backbone
model = smp.Unet(
    encoder_name    = "efficientnet-b4",   # backbone
    encoder_weights = "imagenet",          # pretrained
    in_channels     = 10,                  # your 10-band input
    classes         = 1                    # binary segmentation
)
model.to(device)


# 2) Datasets & loaders (unchanged)
train_dataset = ArcticLabeledImagePatchDataset(
    config.IMAGE_GRIDS_PATH,
    config.LABEL_GRIDS_PATH,
    config.train_grids,
    config.patch_size,
    config.stride
)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

val_dataset = ArcticLabeledImagePatchDataset(
    config.IMAGE_GRIDS_PATH,
    config.LABEL_GRIDS_PATH,
    config.val_grids,
    config.patch_size,
    config.stride
)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


# 3) Loss, optimizer (unchanged)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# 4) Training loop (unchanged)
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for images, labels in train_loader:
        labels = labels.unsqueeze(1).to(device)
        outputs = model(images.to(device))
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # … validation, plotting, saving, etc. …
    print(f"Epoch {epoch}: train_loss={total_train_loss/len(train_loader):.4f}")
