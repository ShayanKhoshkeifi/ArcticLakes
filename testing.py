import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet_model import UNet
from dataloaders import ArcticLabeledImagePatchDataset
from test_comparisons import compute_metrics
from visualize import plot_comparisons
import config
import matplotlib
matplotlib.use('Agg')                        # Stops matplotlib from trying to display figures in a
import matplotlib.pyplot as plt              # non-GUI environment, otherwise we get a warning message
device = 'cuda'



# Sets the specific model that will be used in testing
chosen_epoch = config.test_model


# Retrieves the path to the chosen model
model_path = os.path.join(config.model_dir, f"model_epoch{chosen_epoch}.pt")


# Loads the desired model
print("\nInstantiating model...")
model = UNet(n_channels=10, n_classes=1, bilinear=False)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Complete.\n")


# Creates the testing dataset
print("\nCreating training dataset...")
test_dataset = ArcticLabeledImagePatchDataset(
    config.IMAGE_GRIDS_PATH,
    config.LABEL_GRIDS_PATH,
    config.test_grids,                       # Create subset of grids without rivers
    config.patch_size,
    config.stride
)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)


with torch.no_grad():
    full_precision, full_recall, full_f1, full_iou = 0, 0, 0, 0
    count = 0
    for images, labels in test_loader:
        labels = labels.unsqueeze(1)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # Calculates and adds up metrics for each image
        precision, recall, f1, iou, conf_matrix = compute_metrics(outputs[0][0], labels[0][0])
        full_precision += precision
        full_recall += recall
        full_f1 += f1
        full_iou += iou

        count += 1
        
        
        # Extracts first sample from batch [0] and moves image, label, and output to CPU for plotting
        sample_image = images[0][config.band].cpu().numpy()
        sample_label = labels[0][0].cpu().numpy()
        sample_output = torch.sigmoid(outputs[0][0]).cpu().numpy()             # Normalizes outputs between 0 and 1

        # Prints metrics beneath the plot
        text = f"Precision: {precision:.4f}  -  Recall: {recall:.4f}  -  F1: {f1:.4f}  -  IoU: {iou:.4f}"
        
        # Plots and saves the comparisons for each sample
        fig = plot_comparisons(sample_image, sample_label, sample_output, config.band)
        fig.text(0.5, -0.05, text, ha='center', fontsize=15)
        plt.savefig(f'comparison_plots/testing/{config.experiment_id}/Comparison_Sample_{count}.png', bbox_inches='tight')
        plt.close(fig)

        
    # Calculates and prints averages for all metrics
    avg_precision = full_precision / count
    avg_recall = full_recall / count
    avg_f1 = full_f1 / count
    avg_iou = full_iou / count
    print(f"Precision: {avg_precision:.4f} - Recall: {avg_recall:.4f} - F1: {avg_f1:.4f} - IoU: {avg_iou:.4f}")


    