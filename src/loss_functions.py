import torch
import torch.nn as nn
import torch.nn.functional as F


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
            num_classes = inputs.shape[1] if inputs.shape[1] < inputs.shape[-1] or inputs.shape[1] == inputs.shape[-1] else inputs.shape[-1]
    
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
            mask = targets_flat != self.ignore_index
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
            mask = targets_flat != self.ignore_index
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=None):
        """
        Focal Loss for any type of classification/segmentation task.
        
        Args:
            alpha: Class weights tensor of shape (num_classes,) or None
            gamma: Focusing parameter (default: 2.0)
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
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Standardize input shape to (B*N, C)
        inputs_flat, targets_flat, num_classes = _standardize_inputs(inputs, targets)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets_flat != self.ignore_index
            inputs_flat = inputs_flat[mask]
            targets_flat = targets_flat[mask]
            if inputs_flat.numel() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')

        # Calculate pt (probability of true class)
        pt = torch.exp(-ce_loss)

        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_per_sample = self.alpha.gather(0, targets_flat)
            focal_term = alpha_per_sample * focal_term

        # Combine to get focal loss
        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLossWithArcFaceTransform(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=None, margin = 0.3, scale = 30.0):
        """
        Focal Loss for any type of classification/segmentation task.
        
        Args:
            alpha: Class weights tensor of shape (num_classes,) or None
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Class index to ignore in loss calculation
            margin: m - margins in radius 
            scale: s - scaling factor 
        
        Input shapes supported:
            - Classification: (B, C) with targets (B,)
            - Point clouds: (B, C, N) or (B, N, C) with targets (B, N)
            - Images: (B, C, H, W) with targets (B, H, W)
            - Videos: (B, C, T, H, W) with targets (B, T, H, W)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.margin = margin
        self.scale = scale 

    def forward(self, inputs, targets):
        # Standardize input shape to (B*N, C)
        inputs_flat, targets_flat, num_classes = _standardize_inputs(inputs, targets)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets_flat != self.ignore_index
            inputs_flat = inputs_flat[mask]
            targets_flat = targets_flat[mask]
            if inputs_flat.numel() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # Adding ArcFace Margins to logits 
        inputs_flat = arcface_transform(inputs_flat, targets_flat, self.margin, self.scale)

        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')

        # Calculate pt (probability of true class)
        pt = torch.exp(-ce_loss)

        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_per_sample = self.alpha.gather(0, targets_flat)
            focal_term = alpha_per_sample * focal_term

        # Combine to get focal loss
        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def arcface_transform(inputs_flat, targets_flat, margin, scale):
    """
    Adding ArcFace margins to logits
    
    Args:
        inputs_flat: (B*N, C) - Raw logits
        targets_flat: (B*N,) - True classes
        margin: m - margins in radius 
        scale: s - scaling factor 
    """
    # Standardize input shape to (B*N, C)
    normalized = F.normalize(inputs_flat, p=2, dim=1)  
    
    # Extracting values for true classes
    cos_theta = normalized.gather(1, targets_flat.unsqueeze(1)) 
    cos_theta = cos_theta.squeeze(1)  
    
    # Adding margin: cos(theta + m)
    theta = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    theta_m = theta + margin
    cos_theta_m = torch.cos(theta_m)
    
    # Changing values only for true labels
    normalized.scatter_(1, targets_flat.unsqueeze(1), cos_theta_m.unsqueeze(1))
    
    # Scaling 
    logits_arcface = normalized * scale
    
    return logits_arcface



class LabelSmoothingFocalLoss(nn.Module):
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
        super(LabelSmoothingFocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0.0, 1.0)"
        assert gamma >= 0.0, "Gamma must be non-negative"
    
    def forward(self, inputs, targets):
        # Standardize input shape to (B*N, C)
        inputs_flat, targets_flat, num_classes = _standardize_inputs(inputs, targets)
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets_flat != self.ignore_index
            inputs_flat = inputs_flat[mask]
            targets_flat = targets_flat[mask]
            if inputs_flat.numel() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Get log probabilities and probabilities
        log_probs = F.log_softmax(inputs_flat, dim=-1)
        probs = torch.exp(log_probs)
        
        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(targets_flat, num_classes=num_classes).float()
        
        # Apply label smoothing
        if self.smoothing > 0:
            confidence = 1.0 - self.smoothing
            smooth_value = self.smoothing / (num_classes - 1)
            targets_smooth = targets_one_hot * confidence + smooth_value * (1 - targets_one_hot)
        else:
            targets_smooth = targets_one_hot
        
        # Calculate focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Calculate focal loss with smooth labels
        loss = -targets_smooth * log_probs * focal_weight
        
        # Apply class weights
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.unsqueeze(0).expand(inputs_flat.size(0), -1)
            loss = loss * alpha_t
        
        # Sum over classes
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss