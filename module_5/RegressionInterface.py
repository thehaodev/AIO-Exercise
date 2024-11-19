from abc import ABC, abstractmethod


# Define the interface
class RegressionInterface(ABC):

    @abstractmethod
    def predict(self, x_features):
        """Predicts class probabilities for input data X."""
        pass

    @abstractmethod
    def compute_loss(self, x_features, y_labels):
        """Computes the loss given the input data X and true labels y."""
        pass

    @abstractmethod
    def compute_gradient(self, x_features, y_labels):
        """Computes the gradient of the loss with respect to the model parameters."""
        pass

    @abstractmethod
    def update_gradient(self, x_features, y_labels):
        """Updates the model parameters using the computed gradients."""
        pass
