
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm
import sys
from typing import Optional, Union
import numpy as np
from pathlib import Path
import h5py

def get_Probabilities(logits: torch.Tensor):
    """
    Convert logits to probabilities using softmax.
    
    Args:
        logits (torch.Tensor): Raw model outputs (batch_size, num_classes)
    
    Returns:
        torch.Tensor: Probability distribution over classes
    """
    probs = F.softmax(logits, dim=1)
    return probs

def get_intLabels(probabilities: torch.Tensor):
    """
    Convert probabilities to integer class labels.
    
    Args:
        probabilities (torch.Tensor): Probability distribution over classes
    
    Returns:
        torch.Tensor: Predicted class indices
    """
    labels = torch.argmax(probabilities, dim=1)
    return labels

def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy.
    
    Args:
        outputs: Model predictions (batch_size, num_classes)
        labels: True labels (batch_size)
    
    Returns:
        float: Accuracy as a fraction between 0 and 1
    """
    predicted = torch.argmax(outputs, dim=1)

    correct = (predicted == labels).sum().item()

    # Final number of points
    total = labels.numel()

    accuracy = correct / total
    return accuracy

def calculate_weighted_accuracy(outputs, labels, weights):
    """
    Calculate weighted accuracy where each class has a different weight.
    Only considers classes that are present in the labels (automatic via indexing).
    
    Args:
        outputs: Model predictions (batch_size x num_classes)
        labels: True labels (batch_size)
        weights: Class weights tensor (num_classes) or (1 x num_classes)
    
    Returns:
        Weighted accuracy as a float
    """
    predicted = torch.argmax(outputs, dim=1)
    correct = (predicted == labels).float()
    
    # Normalize weights to [0, 1] range
    weights = weights.squeeze()
    weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    # Get weights for each sample based on their true label
    # This automatically filters to only present labels
    sample_weights = weights_normalized[labels]
    
    # Calculate weighted accuracy
    weighted_correct = (correct * sample_weights).sum().item()
    total_weights = sample_weights.sum().item()
    
    # Avoid division by zero
    if total_weights == 0:
        return 0.0
    
    accuracy = weighted_correct / total_weights
    return accuracy


def get_dataset_len(loader, verbose = False):
    """
    Get the total number of batches in a DataLoader.
    
    Args:
        loader: PyTorch DataLoader
        verbose (bool): Whether to print progress information
    
    Returns:
        int: Total number of batches in the loader
    """
    total = 0
    if verbose:
        print('\nGetting dataset size...\n')
    for _ in enumerate(loader):

        if total%10==0 and verbose:
            sys.stdout.write(f"\rProcessing iteration: {total}")
            sys.stdout.flush()

        total += 1
    if verbose:
        sys.stdout.write(f"\n\rProcessing iteration: {total}\n")
        sys.stdout.flush()

    return total

def compute_pos_weights_h5(h5_path: Union[str, Path],
                        num_classes: int,
                        power: float = 0.25) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            cloud = f[key][:]

            labels = cloud[..., -1].astype(np.int32).ravel()

            counts += np.bincount(labels, minlength=num_classes)

    weights              = (1.0 / (counts + 1e-6)) ** power
    weights[counts == 0] = 0.0
    weights              = (weights / weights.max()).astype(np.float32)
    return torch.from_numpy(weights)

def compute_pos_weights_cloud(data_dir, num_classes: int,
                        power: float = 0.25) -> torch.Tensor:
    """
    Inverse frequency weights with power dampening.
    power=1.0 → raw inverse freq
    power=0.5 → sqrt dampening
    power=0.25 → fourth root (default, mild compression)
    power=0.0 → uniform
    """


    counts = np.zeros(num_classes, dtype=np.int64)
    for path in sorted(Path(data_dir).glob("*.npy")):
        labels  = np.load(path)[:, 4].astype(np.int32)
        counts += np.bincount(labels, minlength=num_classes)

    weights              = (1.0 / (counts + 1e-6)) ** power
    weights[counts == 0] = 0.0
    weights              = (weights / weights.max()).astype(np.float32)

    weights = torch.from_numpy(weights)
    return weights

def compute_pos_weights(data_dir, num_classes: int,
                        power: float = 0.25, ignore_index: Optional[int] = None) -> torch.Tensor:
    """
    Inverse frequency weights with power dampening.
    power=1.0 → raw inverse freq
    power=0.5 → sqrt dampening
    power=0.25 → fourth root (default, mild compression)
    power=0.0 → uniform
    """

    counts = np.zeros(num_classes, dtype=np.int64)
    for path in sorted(Path(data_dir).glob("*.npy")):
        print(f"Processing file: {path.name}")
        labels = int(path.stem.rsplit('_', 1)[-1])
        if ignore_index is not None:
            if labels == ignore_index:
                continue
        counts[labels] += 1

    print(f"Class counts: {counts}")
  
    weights              = (1.0 / (counts + 1e-6)) ** power
    weights[counts == 0] = 0.0
    weights              = (weights / weights.max()).astype(np.float32)

    weights = torch.from_numpy(weights)

    labels = [int(p.stem.rsplit('_', 1)[-1]) for p in sorted(Path(data_dir).glob("*.npy"))]
    if ignore_index is not None:
        labels = [l for l in labels if l != ignore_index]

    return weights, labels


def compute_mIoU(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    Compute mean Intersection over Union (mIoU) for segmentation tasks.
    
    Args:
        predictions (torch.Tensor): Model predictions (can be logits or class indices)
        targets (torch.Tensor): Ground truth labels
        num_classes (int): Total number of classes
    
    Returns:
        tuple: (mean_iou, class_ious) where mean_iou is float and class_ious is tensor
    """
    if predictions.dim() > targets.dim():
        # If predictions are logits/ probs (with class dimension), convert to class indices
        predictions = torch.argmax(predictions, dim=1)

    # Ensure inputs are on the same device
    if predictions.device != targets.device:
        predictions = predictions.to(targets.device)

    # Flatten the tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Initialize IoU for each class
    class_ious = torch.zeros(num_classes, device=targets.device)

    # Compute IoU for each class
    for class_idx in range(num_classes):
        # True Positives: prediction and target are both class_idx
        pred_inds = predictions == class_idx
        target_inds = targets == class_idx

        # Intersection and union
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        # Compute IoU for this class (handle division by zero)
        if union > 0:
            class_ious[class_idx] = intersection / union

    # Compute mean IoU across classes that appear in the targets
    valid_classes = torch.unique(targets)
    if len(valid_classes) == 0:
        return 0.0, class_ious

    valid_ious = torch.index_select(class_ious, 0, valid_classes)
    miou = valid_ious.mean().item()

    return miou, class_ious


