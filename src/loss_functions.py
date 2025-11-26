class IoULoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, reduction='mean', ignore_index=None):
        super(IoULoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth # A small value to avoid division by zero
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (BatchSize, NumClasses, NumPoints) - raw logits
        # targets: (BatchSize, NumPoints) - class IDs

        # Apply softmax to get probabilities
        # Transpose inputs to (BatchSize, NumPoints, NumClasses) for one-hot
        inputs = F.softmax(inputs, dim=1).permute(0, 2, 1) # (B, N, C)

        # Reshape for easier calculation
        inputs = inputs.contiguous().view(-1, self.num_classes) # (B*N, C)
        targets = targets.contiguous().view(-1) # (B*N)

        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            inputs = inputs[mask]
            targets = targets[mask]
            if inputs.numel() == 0: # If all points are ignored
                return torch.tensor(0.0, device=inputs.device)

        # Convert targets to one-hot encoding
        # (B*N) -> (B*N, NumClasses)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Intersection: (B*N, C) * (B*N, C) -> (B*N, C) then sum over B*N -> (C)
        intersection = (inputs * targets_one_hot).sum(dim=0) # Sum over all points for each class

        # Union: (B*N, C) + (B*N, C) - (B*N, C) -> (B*N, C) then sum over B*N -> (C)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0) - intersection

        # IoU for each class
        iou = (intersection + self.smooth) / (union + self.smooth)

        # IoU Loss: 1 - IoU
        loss = 1.0 - iou

        if self.reduction == 'mean':
            return loss.mean() # Average over all classes
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none' or 'elementwise'
            return loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, class_weights = None, reduction='mean', ignore_index=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth # A small value to avoid division by zero
        self.class_weights = class_weights
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (BatchSize, NumClasses, NumPoints) - raw logits
        # targets: (BatchSize, NumPoints) - class IDs

        # Apply softmax to get probabilities
        # Transpose inputs to (BatchSize, NumPoints, NumClasses) for one-hot
        inputs = F.softmax(inputs, dim=1).permute(0, 2, 1) # (B, N, C)

        # Reshape for easier calculation
        inputs = inputs.contiguous().view(-1, self.num_classes) # (B*N, C)
        targets = targets.contiguous().view(-1) # (B*N)

        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            inputs = inputs[mask]
            targets = targets[mask]
            if inputs.numel() == 0: # If all points are ignored
                return torch.tensor(0.0, device=inputs.device)

        # Convert targets to one-hot encoding
        # (B*N) -> (B*N, NumClasses)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Intersection: (B*N, C) * (B*N, C) -> (B*N, C) then sum over B*N -> (C)
        intersection = (inputs * targets_one_hot).sum(dim=0) # Sum over all points for each class

        # Sum of elements in prediction and target
        # Sum over B*N for each class
        sum_pred = inputs.sum(dim=0)
        sum_target = targets_one_hot.sum(dim=0)

        # Dice coefficient for each class
        dice = (2. * intersection + self.smooth) / (sum_pred + sum_target + self.smooth)

        # Dice Loss: 1 - Dice
        loss = 1.0 - dice

        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device)

        if self.reduction == 'mean':
            return loss.mean() # Average over all classes
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none' or 'elementwise'
            return loss

class FocalLoss_CLASS(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where:
    - p_t is the probability for the true class
    - alpha is the class balance weight (optional)
    - gamma is the focusing parameter (typically 2.0)
    
    Args:
        alpha (float or Tensor): Class balance weight. If float, applies the same weight
                                 to all classes. If Tensor, must have length equal to
                                 the number of classes.
        gamma (float): Focusing parameter. Higher values focus more on hard examples.
                       Default is 2.0.
        reduction (str): 'none' | 'mean' | 'sum'. Default is 'mean'.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss_CLASS, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # If alpha is a list or tensor, convert to tensor
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes) - raw logits
            targets: Tensor of shape (batch_size,) or (batch_size, 1) - class indices
        
        Returns:
            loss: Scalar tensor (if reduction='mean' or 'sum') or tensor (batch_size,)
        """
        
        # Ensure targets has dimensions (batch_size,)
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        # Calculate cross entropy loss without reduction
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Calculate probabilities for the focal weight
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal Loss: focal_weight * ce_loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Ensure alpha is on the same device as logits
                if self.alpha.device != logits.device:
                    self.alpha = self.alpha.to(logits.device)
                
                # Select appropriate alpha weights for each example
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
        

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=None, device = torch.device('cpu')):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (BatchSize, NumClasses, NumPoints) - raw logits
        # targets: (BatchSize, NumPoints) - class IDs

        if self.ignore_index is not None:
            # Create a mask for valid (non-ignored) points
            mask = (targets != self.ignore_index)
            # Apply mask to targets to get only relevant labels
            targets_masked = targets[mask]
            # Apply mask to inputs. This needs careful reshaping.
            # outputs needs to be (B*N, C) before masking.
            inputs_reshaped = inputs.permute(0, 2, 1).contiguous().view(-1, inputs.shape[1])
            inputs_masked = inputs_reshaped[mask]

            if inputs_masked.numel() == 0:
                return torch.tensor(0.0, device=inputs.device)
        else:
            inputs_reshaped = inputs.permute(0, 2, 1).contiguous().view(-1, inputs.shape[1])
            inputs_masked = inputs_reshaped
            targets_masked = targets.contiguous().view(-1)

        # Step 1: Calculate Cross-Entropy Loss
        # F.cross_entropy expects inputs as (N, C) and targets as (N)
        ce_loss = F.cross_entropy(inputs_masked, targets_masked, reduction='none')

        # Step 2: Calculate pt (probability of the true class)
        # Apply softmax to logits to get probabilities
        # Then use gather to pick probabilities corresponding to the true class
        pt = torch.exp(-ce_loss) # Alternatively: pt = F.softmax(inputs_masked, dim=1).gather(1, targets_masked.view(-1, 1))

        # Step 3: Calculate the modulating factor (1 - pt)^gamma
        focal_term = (1 - pt)**self.gamma

        # Step 4: Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device: # Ensure alpha is on the correct device
                self.alpha = self.alpha.to(inputs.device)

            # Get alpha for each target class
            alpha_per_sample = self.alpha.gather(0, targets_masked)
            weighted_focal_term = alpha_per_sample * focal_term
        else:
            weighted_focal_term = focal_term

        # Step 5: Combine everything to get Focal Loss
        loss = weighted_focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss


class LabelSmoothingFocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for imbalanced classification.
    
    Combines:
    - Focal Loss: Focuses on hard examples by down-weighting easy ones
    - Label Smoothing: Prevents overconfidence and improves generalization
    - Class Weights: Handles class imbalance
    
    Args:
        alpha: Class weights tensor of shape (num_classes,) or None
        gamma: Focusing parameter for focal loss (default: 2.0)
               Higher gamma = more focus on hard examples
        smoothing: Label smoothing parameter (default: 0.1)
                   smoothing=0.0 means no smoothing
        reduction: 'mean', 'sum', or 'none'
    
    Example:
        >>> # For 10 classes with class weights
        >>> weights = torch.tensor([1.0, 1.5, 2.0, ...])  # 10 values
        >>> criterion = LabelSmoothingFocalLoss(alpha=weights, gamma=2.0, smoothing=0.1)
        >>> 
        >>> # During training
        >>> outputs = model(inputs)  # Shape: (batch_size, num_classes)
        >>> loss = criterion(outputs, targets)  # targets shape: (batch_size,)
    """
    
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingFocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        
        # Validate parameters
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0.0, 1.0)"
        assert gamma >= 0.0, "Gamma must be non-negative"
        assert reduction in ['mean', 'sum', 'none'], "Invalid reduction mode"
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions of shape (batch_size, num_classes) - logits (before softmax)
            targets: Ground truth labels of shape (batch_size,) - class indices
        
        Returns:
            Loss value (scalar if reduction='mean' or 'sum', tensor if 'none')
        """
        batch_size = inputs.size(0)
        num_classes = inputs.size(1)
        
        # Get log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Apply label smoothing
        if self.smoothing > 0:
            # Smooth labels: (1 - smoothing) for true class, smoothing/(num_classes-1) for others
            confidence = 1.0 - self.smoothing
            smooth_value = self.smoothing / (num_classes - 1)
            
            targets_smooth = targets_one_hot * confidence + smooth_value * (1 - targets_one_hot)
        else:
            targets_smooth = targets_one_hot
        
        # Calculate focal weight: (1 - p_t)^gamma
        # For each sample, get the probability of the true class
        focal_weight = (1 - probs) ** self.gamma
        
        # Calculate focal loss with smooth labels
        # Loss = -alpha * (1-p)^gamma * sum(smooth_label * log(p))
        loss = -targets_smooth * log_probs * focal_weight
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Expand alpha to match batch size
            alpha_t = self.alpha.unsqueeze(0).expand(batch_size, -1)
            loss = loss * alpha_t
        
        # Sum over classes
        loss = loss.sum(dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss