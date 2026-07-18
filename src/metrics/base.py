from abc import ABC, abstractmethod
import numpy as np

class BaseMetric(ABC):
    @abstractmethod
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the metric based on true and predicted values."""
        return
    
    