import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from unet_model import UNet
from efficient_unet import UNetEfficientNet
from dataloaders import ArcticLabeledImagePatchDataset
from visualize import plot_comparisons
import config
import matplotlib
matplotlib.use('Agg')                        # Stops matplotlib from trying to display figures in a
import matplotlib.pyplot as plt              # non-GUI environment, otherwise we get a warning message
device = 'cuda'



# Instantiates model
print("\nInstantiating model...")
model = UNet(
    n_channels=10,                           # 10: number of bands in the satellite images
    n_classes=1,                             # 1: binary segmentation -- either water or land
    bilinear=False                           # True: bilinear upsampling/interpolation // False: transposed convolution
)
model.to(device)
print("Complete.")




# Loads the training dataset
print("\nLoading training dataset...")
train_dataset = ArcticLabeledImagePatchDataset(
    config.IMAGE_GRIDS_PATH,
    config.LABEL_GRIDS_PATH,
    config.train_grids,
    config.patch_size,
    config.stride
)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)


# Loads the validation dataset
print("\nLoading validation dataset...")
val_dataset = ArcticLabeledImagePatchDataset(
    config.IMAGE_GRIDS_PATH,
    config.LABEL_GRIDS_PATH,
    config.val_grids,
    config.patch_size,
    config.stride
)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)




# Defines loss using BCEWithLogitsLoss, which combines a sigmoid layer and binary cross-entropy loss in one class
criterion = nn.BCEWithLogitsLoss()

# Defines optimizer using PyTorch's Adam algorithm with learning rate of 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Defines scheduler that reduces LR as val loss stops improving
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) if config.scheduler_on else None



train_loss_array = []                          # List of train loss for each epoch
val_loss_array = []                            # List of val loss for each epoch
total_duration = 0                             # Total duration of training loop

for epoch in range(config.num_epochs):
    start_time = time.time()                   # Starts timing each epoch
    
    torch.cuda.empty_cache()                   # Empties the cache to avoid hitting GPU memory limitations
    model.train()                              # Puts the model into "training mode"
    count = 0
    total_train_loss = 0

    # Training loop
    for images, labels in train_loader:
        labels = labels.unsqueeze(1)           # Adds channel dimension to make it the same shape as images (B, C, H, W)
        count += 1
        
        # Forward pass
        outputs = model(images.to(device))

        # Loss calculation (BCE)
        train_loss = criterion(outputs, labels.to(device))
        total_train_loss += train_loss

        # Backward pass
        optimizer.zero_grad()                  # Clears the optimizer of previous iterations' gradients
        train_loss.backward()                  # Performs backpropagations
        optimizer.step()                       # Updates the optimizer with new gradients
        
    # Generates training loss for model
    avg_train_loss = total_train_loss / count


    
    with torch.no_grad():                      # Disables gradient tracking - helps avoid hitting GPU memory limitations
        model.eval()                           # Puts the model into "validation mode"
        count = 0
        total_val_loss = 0
        
        # Validation loop
        for images, labels in val_loader:
            labels = labels.unsqueeze(1)
            count += 1
            
            outputs = model(images.to(device))
            
            # Loss calculation
            val_loss = criterion(outputs, labels.to(device))
            total_val_loss += val_loss


            # Creates plots for only a select number of images during each epoch
            if count <= config.num_visual_samples:
    
                # Extracts first sample from batch [0] and moves image, label, and output to CPU for plotting
                sample_image = images[0][config.band].cpu().numpy()
                sample_label = labels[0][0].cpu().numpy()
                sample_output = torch.sigmoid(outputs[0][0]).cpu().numpy()             # Normalizes outputs between 0 and 1

                # Plots and saves the comparisons for each epoch and sample
                fig = plot_comparisons(sample_image, sample_label, sample_output, config.band)
                plt.savefig(f'comparison_plots/validation/{config.experiment_id}/Comparison_Epoch_{epoch}_Sample_{count}.png')
                plt.close(fig)

            
        # Generates validation loss for each model
        avg_val_loss = total_val_loss / count

        # Adjusts learning rate if scheduler != None
        scheduler and scheduler.step(avg_val_loss)
    
        # Prints the last learning rate used for each epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} - Learning rate: {current_lr}")


    
    # Prints training & validation loss for every epoch
    print(f"Epoch {epoch} — Training loss: {avg_train_loss.item():.4f} - Validation Loss: {avg_val_loss.item():.4f}")

    # Adds each epoch's train & val loss to the appropriate array for future plotting
    train_loss_array.append(avg_train_loss.item())
    val_loss_array.append(avg_val_loss.item())

    # Saves the model generated during each epoch
    torch.save(model.state_dict(), os.path.join(config.model_dir, f"model_epoch{epoch}.pt"))

    # Calculates and prints epoch's duration
    end_time = time.time()
    duration = end_time - start_time
    total_duration += duration
    print(f"Epoch {epoch} - Duration: {duration:.2f} seconds")

    # Separates each epoch
    print("=======")

# Calculates an average training speed
train_speed = total_duration / config.num_epochs

# Generates line graphs of train & val loss over all epochs
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
training_loss = np.array(train_loss_array)
validation_loss = np.array(val_loss_array)
plt.tight_layout(pad=3.0)

# Plots the training loss
axs[0].plot(training_loss)
axs[0].set_title('Training Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

# Plots the validation loss
axs[1].plot(validation_loss)
axs[1].set_title('Validation Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].grid(True)

# Prints model settings beneath the plot
text = f"Experiment: {config.experiment_id}  -  Avg Training Speed: {train_speed:.1f}s"
fig.text(0.5, -0.05, text, ha='center', fontsize=15)

# Saves the loss plots
plt.savefig(f'loss_plots/{config.experiment_id.lower()}_loss_plot.png', bbox_inches='tight')
