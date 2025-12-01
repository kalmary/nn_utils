import torch

class EarlyStopping:
    """
    Early stopping callback to prevent overfitting during training.
    
    Args:
        patience (int): Number of epochs to wait before stopping after no improvement
        delta (float): Minimum change in loss to qualify as an improvement
        verbose (bool): Whether to print early stopping messages
    """
    def __init__(self, patience=5, delta=0.001, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        """
        Check if training should be stopped early based on validation loss.
        
        Args:
            val_loss (float): Current validation loss
        
        Returns:
            None: Updates internal state and sets stop_training flag if needed
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")