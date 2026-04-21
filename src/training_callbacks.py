from typing import Literal

class EarlyStopping:
    """
    Early stopping callback to prevent overfitting during training.
    
    Args:
        patience (int): Number of epochs to wait before stopping after no improvement
        delta (float): Minimum change in loss to qualify as an improvement
        verbose (bool): Whether to print early stopping messages
    """
    def __init__(self, patience=5, delta=0.001, mode: Literal["maximize", "minimize"] = "minimize", verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_val= None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val):
        """
        Check if training should be stopped early based on validation loss.
        
        Args:
            val_loss (float): Current validation loss/metric
        
        Returns:
            None: Updates internal state and sets stop_training flag if needed
        """
        improved = (
            self.best_val is None or
            (self.mode == "minimize" and val < self.best_val - self.delta) or
            (self.mode == "maximize" and val > self.best_val + self.delta)
        )
        if improved:
            self.best_val = val
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")