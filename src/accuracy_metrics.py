
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm
import sys
from typing import Optional

def get_Probabilities(logits: torch.Tensor):
    probs = F.softmax(logits, dim=1)
    return probs

def get_intLabels(probabilities: torch.Tensor):
    labels = torch.argmax(probabilities, dim=1)
    return labels

def calculate_accuracy(outputs, labels):

    """

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

def calculate_class_weights_classification(loader, num_classes, total=None, device='cpu'):
    class_sample_counts = Counter()
    total_samples = 0

    print("\nClass weights computation for classification...")
    progressbar = tqdm(enumerate(loader), desc='Getting class weights from dataset', total=total)

    for i, batch in progressbar:

        targets_np = batch.y.cpu().numpy()
        class_sample_counts.update(targets_np)  # No need to flatten, targets are already sample-wise

        total_samples += targets_np.shape[0]  # targets_np.size also works, but shape[0] is more explicit for samples

    weights = torch.zeros(num_classes, device=device)
    for class_idx in range(num_classes):
        count = class_sample_counts.get(class_idx, 0)  # Use .get to handle classes not present in the batch

        if count == 0:
            print(
                f"Warning: Class {class_idx} has no samples in the dataset. Its weight will be 0 before normalization.")

        else:
            weights[class_idx] = total_samples / (count * num_classes)

    if weights.sum() == 0:
        raise ValueError("All class counts are zero, cannot compute weights. Check your dataset and num_classes.")
    return weights / weights.sum() * num_classes


def calculate_class_weights(loader: torch.utils.data.DataLoader,
                            num_classes,
                            total: Optional[int] = None, 
                            device: str ='cpu',
                            verbose: bool = True) -> torch.Tensor:
    """
    Calculates weights for each class (meant for unbalanced datasets)
    """
    class_pixel_counts = Counter()
    total_pixels = 0

    if verbose:
        print("\nObliczanie wag klas...")
    
    progressbar = enumerate(loader)


    for i, (_, targets) in progressbar:
        # move labels to cpu
        targets_np = targets.cpu().numpy().flatten()

        class_pixel_counts.update(targets_np)
        total_pixels += targets_np.size

        if verbose:
            if i % 10 == 0:
                sys.stdout.write(f"\rProcessing iteration: {i}/{total}")
                sys.stdout.flush()
    if verbose:
        sys.stdout.write(f"\n\rProcessing iteration: {i}/{total}\n")
        sys.stdout.flush()

    weights = torch.zeros(num_classes, device=device)
    for class_idx in range(num_classes):
        # if no class in dataset then use 0 value
        count = class_pixel_counts.get(class_idx, 0)

        if count == 0 and verbose:
            print(f"Klasa {class_idx} nie ma pikseli w zbiorze danych.")
        else:
            # weights normalization
            weights[class_idx] = total_pixels / (count * num_classes)

    # final normalization
    return weights / weights.sum() * num_classes


def compute_mIoU(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):

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


