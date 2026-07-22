from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions using the fitted model."""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Evaluate the model's performance."""
        pass

    @abstractmethod
    def save(self, file_path):
        """Save the model to a file."""
        pass

    @abstractmethod
    def load(self, file_path):
        """Load the model from a file."""
        pass