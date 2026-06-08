import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


def _standardize_inputs(inputs, targets, num_classes=None):
    """
    Standardize inputs to (B*N, C) and targets to (B*N) for any dimensionality.
    
    Handles:
    - Classification: (B, C) -> targets (B,)
    - Point clouds: (B, C, N) or (B, N, C) -> targets (B, N)
    - Images: (B, C, H, W) -> targets (B, H, W)
    - Videos: (B, C, T, H, W) -> targets (B, T, H, W)
    
    Args:
        inputs: Logits tensor
        targets: Ground truth tensor
        num_classes: Number of classes (optional, will be inferred if None)
    
    Returns:
        logits: (B*N, C) where N is product of all spatial dimensions
        targets: (B*N,)
        num_classes: Inferred or provided number of classes
    """
    # Infer num_classes if not provided
    if num_classes is None:
        if inputs.dim() == 2:
            num_classes = inputs.shape[1]
        else:
            # Check if dim=1 looks like channels (typically smaller than spatial dims)
            num_classes = inputs.shape[1] if inputs.shape[1] <= inputs.shape[-1] else inputs.shape[-1]    

    if inputs.dim() == 2:
        # Regular classification: (B, C)
        logits = inputs
        
    elif inputs.dim() >= 3:
        # Check if dim=1 is the channel dimension
        if inputs.shape[1] == num_classes:
            # Channel-first format: (B, C, N) or (B, C, H, W) or (B, C, T, H, W)
            # Move channel to last: (B, ..., C)
            dims_order = [0] + list(range(2, inputs.dim())) + [1]
            logits = inputs.permute(*dims_order)
        else:
            # Already channel-last: (B, N, C) or (B, H, W, C) or (B, T, H, W, C)
            logits = inputs
    else:
        raise ValueError(f"Expected input with at least 2 dimensions, got {inputs.dim()}D")
    
    # Flatten to (B*N, C)
    logits = logits.contiguous().view(-1, num_classes)
    targets = targets.contiguous().view(-1)
    
    return logits, targets, num_classes


class IoULoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, reduction='mean', ignore_index=None):
        """
        IoU Loss for any type of segmentation task.
        
        Args:
            num_classes: Number of classes
            smooth: Smoothing factor to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Class index to ignore in loss calculation
        
        Input shapes supported:
            - Classification: (B, C) with targets (B,)
            - Point clouds: (B, C, N) or (B, N, C) with targets (B, N)
            - Images: (B, C, H, W) with targets (B, H, W)
            - Videos: (B, C, T, H, W) with targets (B, T, H, W)
        """
        super(IoULoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Standardize input shape to (B*N, C)
        inputs_flat, targets_flat, num_classes = _standardize_inputs(inputs, targets, self.num_classes)
        
        # Apply softmax to get probabilities
        probs = F.softmax(inputs_flat, dim=1)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets_flat < self.ignore_index
            probs = probs[mask]
            targets_flat = targets_flat[mask]
            if probs.numel() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets_flat, num_classes=num_classes).float()

        # Calculate intersection and union per class
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0) - intersection

        # IoU for each class
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, class_weights=None, reduction='mean', ignore_index=None):
        """
        Dice Loss for any type of segmentation task.
        
        Args:
            num_classes: Number of classes
            smooth: Smoothing factor to avoid division by zero
            class_weights: Tensor of shape (num_classes,) for class weighting
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Class index to ignore in loss calculation
        
        Input shapes supported:
            - Classification: (B, C) with targets (B,)
            - Point clouds: (B, C, N) or (B, N, C) with targets (B, N)
            - Images: (B, C, H, W) with targets (B, H, W)
            - Videos: (B, C, T, H, W) with targets (B, T, H, W)
        """
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Standardize input shape to (B*N, C)
        inputs_flat, targets_flat, num_classes = _standardize_inputs(inputs, targets, self.num_classes)
        
        # Apply softmax to get probabilities
        probs = F.softmax(inputs_flat, dim=1)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets_flat < self.ignore_index
            probs = probs[mask]
            targets_flat = targets_flat[mask]
            if probs.numel() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets_flat, num_classes=num_classes).float()

        # Calculate intersection and sums per class
        intersection = (probs * targets_one_hot).sum(dim=0)
        sum_pred = probs.sum(dim=0)
        sum_target = targets_one_hot.sum(dim=0)

        # Dice coefficient for each class
        dice = (2. * intersection + self.smooth) / (sum_pred + sum_target + self.smooth)
        loss = 1.0 - dice

        # Apply class weights
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ArcFaceFocalLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma: float = 2.0,
                 smoothing: float = 0.0,
                 reduction: str = 'mean',
                 ignore_index: Optional[int] = None,
                 margin: float = 0.3,
                 scale: float = 30.0):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.reduction    = reduction
        self.ignore_index = ignore_index
        self.margin       = margin
        self.scale        = scale

        self.ce_loss = nn.CrossEntropyLoss(
            weight=None,
            reduction='none',
            ignore_index=ignore_index if ignore_index is not None else -100,
            label_smoothing=smoothing
        )

    @staticmethod
    def _arcface_logits(embeddings: torch.Tensor,
                        weight: torch.Tensor,
                        targets: torch.Tensor,
                        margin: float,
                        scale: float) -> torch.Tensor:
        emb = F.normalize(embeddings, p=2, dim=1)
        w   = F.normalize(weight, p=2, dim=1)
        cos_theta = emb @ w.T

        theta     = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        cos_theta_m = torch.cos(theta + margin)

        one_hot = torch.zeros_like(cos_theta).scatter_(1, targets.unsqueeze(1), 1.0)
        logits  = scale * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
        return logits

    def forward(self,
                embeddings: torch.Tensor,
                weight: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is not None:
            mask       = targets < self.ignore_index
            embeddings = embeddings[mask]
            targets    = targets[mask]
            if embeddings.numel() == 0:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        logits  = self._arcface_logits(embeddings, weight, targets, self.margin, self.scale)
        ce      = self.ce_loss(logits, targets)
        pt      = torch.exp(-ce)
        focal   = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != embeddings.device:
                self.alpha = self.alpha.to(embeddings.device)
            focal = self.alpha.gather(0, targets) * focal

        loss = focal * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean', ignore_index=None):
        """
        Focal Loss with Label Smoothing for any type of classification/segmentation task.
        
        Args:
            alpha: Class weights tensor of shape (num_classes,) or None
            gamma: Focusing parameter (default: 2.0)
            smoothing: Label smoothing parameter (default: 0.1)
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Class index to ignore in loss calculation
        
        Input shapes supported:
            - Classification: (B, C) with targets (B,)
            - Point clouds: (B, C, N) or (B, N, C) with targets (B, N)
            - Images: (B, C, H, W) with targets (B, H, W)
            - Videos: (B, C, T, H, W) with targets (B, T, H, W)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Create CrossEntropyLoss with label smoothing and no reduction
        self.ce_loss = nn.CrossEntropyLoss(
            weight=None,
            reduction='none',
            ignore_index=ignore_index if ignore_index is not None else -100,
            label_smoothing=smoothing
        )

    def forward(self, inputs, targets):
        # Standardize input shape to (B*N, C)
        inputs_flat, targets_flat, _ = _standardize_inputs(inputs, targets)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets_flat < self.ignore_index
            if not mask.any():
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            inputs_flat = inputs_flat[mask]
            targets_flat = targets_flat[mask]

        # Calculate cross-entropy loss with label smoothing
        ce_loss = self.ce_loss(inputs_flat, targets_flat)

        # Calculate pt (probability of true class) from softmax
        probs = F.softmax(inputs_flat, dim=-1)
        pt = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply focal weighting
        loss = focal_term * ce_loss

        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_per_sample = self.alpha.gather(0, targets_flat)
            loss = alpha_per_sample * loss

        # Apply reduction
        if self.ignore_index is not None:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        
        return loss

class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        """
        Discriminative Loss for instance segmentation feature learning.
        Memory efficient - only computes instance means, not pairwise distances.
        
        Args:
            delta_v: Variance margin (pull same-instance points together)
            delta_d: Distance margin (push different-instance means apart)
            alpha: Weight for variance term
            beta: Weight for distance term  
            gamma: Weight for regularization term
            
        Input shapes:
            features: (B, C, N) - feature embeddings per point
            labels: (B, N) - instance IDs per point
        """
        super(DiscriminativeLoss, self).__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, features, labels):
        """
        Args:
            features: (B, C, N) - C-dimensional features for N points
            labels: (B, N) - instance labels
        """
        B, C, N = features.shape
        
        # Transpose to (B, N, C) for easier processing
        features = features.transpose(1, 2).contiguous()  # (B, N, C)
        
        total_loss = 0.0
        
        for b in range(B):
            feat = features[b]  # (N, C)
            label = labels[b]   # (N,)
            
            unique_labels = torch.unique(label)
            n_instances = len(unique_labels)
            
            if n_instances == 0:
                continue
                
            # Compute instance means
            means = []
            for inst_id in unique_labels:
                mask = (label == inst_id)
                if mask.sum() == 0:
                    continue
                inst_feat = feat[mask]  # (n_points_in_instance, C)
                mean = inst_feat.mean(dim=0)  # (C,)
                means.append(mean)
            
            if len(means) == 0:
                continue
                
            means = torch.stack(means)  # (n_instances, C)
            
            # Variance term: pull points to their instance mean
            var_loss = 0.0
            for idx, inst_id in enumerate(unique_labels):
                mask = (label == inst_id)
                if mask.sum() == 0:
                    continue
                inst_feat = feat[mask]  # (n_points, C)
                mean = means[idx]  # (C,)
                
                # Distance from mean, clamped by margin
                dist = torch.norm(inst_feat - mean, dim=1)  # (n_points,)
                dist = torch.clamp(dist - self.delta_v, min=0.0) ** 2
                var_loss += dist.mean()
            
            var_loss /= n_instances
            
            # Distance term: push instance means apart
            dist_loss = 0.0
            if n_instances > 1:
                # Pairwise distances between means
                for i in range(n_instances):
                    for j in range(i + 1, n_instances):
                        dist = torch.norm(means[i] - means[j])
                        dist = torch.clamp(2 * self.delta_d - dist, min=0.0) ** 2
                        dist_loss += dist
                
                dist_loss /= (n_instances * (n_instances - 1) / 2)
            
            # Regularization term: keep means near origin
            reg_loss = torch.norm(means, dim=1).mean()
            
            # Combine losses
            total_loss += self.alpha * var_loss + self.beta * dist_loss + self.gamma * reg_loss
        
        return total_loss / B


def calculate_l1_penalty_best_practice(model, l1_lambda, device  = torch.device):
    """
    Calculates the L1 penalty (LASSO) efficiently in PyTorch, 
    ignoring bias parameters as per best practice.
    
    Args:
        model (nn.Module): The PyTorch model.
        l1_lambda (float): The regularization hyperparameter (lambda).
        
    Returns:
        torch.Tensor: The calculated L1 penalty term, residing on the 
                      same device as the model weights.
    """
    
    model = model.to(device)

    l1_regularization_list = [
        torch.abs(param).view(-1).to(device)
        for name, param in model.named_parameters()
        if param.requires_grad and 'bias' not in name
    ]
    
    if not l1_regularization_list:
        if model.parameters():
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = 'cpu'
        else:
            device = 'cpu'

        return torch.tensor(0.0, device=device)
    
    del model
    
    all_l1_params = torch.cat(l1_regularization_list)

    l1_norm = torch.sum(all_l1_params)

    return l1_lambda * l1_norm
