from .base import BaseMetric
import pandas as pd
import numpy as np


class ExpectedCalibrationError(BaseMetric):
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def compute_metrics(self, y_true, y_pred):
        """Compute the Expected Calibration Error (ECE) for binary classification."""
        # Create bins        bins = np.linspace(0, 1, self.n_bins + 1)
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1  # Get bin indices for each prediction  
        ece = 0.0
        for i in range(self.n_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_true = y_true[bin_mask]
                bin_pred = y_pred[bin_mask]
                bin_accuracy = np.mean(bin_true)
                bin_confidence = np.mean(bin_pred)
                ece += np.abs(bin_accuracy - bin_confidence) * (np.sum(bin_mask) / len(y_true))
        return ece