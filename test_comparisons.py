import torch
import numpy as np

def compute_metrics(y_pred, y_true, threshold=0.5):
    # Convert logits/probs to binary mask
    y_pred_bin = (y_pred > threshold).float()

    # Flatten all pixels
    y_pred_flat = y_pred_bin.view(-1)
    y_true_flat = y_true.view(-1)

    # Calculate confusion matrix
    TP = torch.sum((y_pred_flat == 1) & (y_true_flat == 1)).item()
    TN = torch.sum((y_pred_flat == 0) & (y_true_flat == 0)).item()
    FP = torch.sum((y_pred_flat == 1) & (y_true_flat == 0)).item()
    FN = torch.sum((y_pred_flat == 0) & (y_true_flat == 1)).item()
    confusion_matrix = np.array([[TP, FP],[FN, TN]])

    # Calculates IoU
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    # Avoid division by zero
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, iou, confusion_matrix
