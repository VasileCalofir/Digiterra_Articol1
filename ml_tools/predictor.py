import pandas as pd
from typing import Any

from ml_tools.trainer import Trainer

class Predictor:
    """
    The Predictor class is designed to generate predictions based on a trained model.
    
    Attributes:
        trainer (Trainer): An instance of the Trainer class containing a trained model and data.
        prediction_on_test (pd.Series): A Pandas Series object to store the predictions on test data.
        prediction_on_train (pd.Series): A Pandas Series object to store the predictions on training data.
    """
    
    def __init__(self, trainer: Trainer):
        """
        Initialize the Predictor class.
        
        Parameters:
            trainer (Trainer): An instance of the Trainer class containing a trained model and data.
        
        Raises:
            ValueError: If the trainer instance does not contain a trained model.
        """
        # Ensure the trainer instance contains a trained model before proceeding.
        if trainer.trained_model is None:
            raise ValueError("Model not set!")
        
        self.trainer = trainer  # Store the trainer instance.
        
        # Initialize attributes to store predictions on test and train data.
        self.prediction_on_test = pd.Series(dtype=float)
        self.prediction_on_train = pd.Series(dtype=float)
        
    def set_prediction_on_test(self):
        """
        Generate and set the predictions on test data.
        
        Raises:
            ValueError: If the test data is not set in the trainer instance.
        """
        # Check if test data is available.
        if self.trainer.X_test is None:
            raise ValueError("Test data not set!")
        
        # Generate predictions on test data and store it.
        self.prediction_on_test = self.trainer.trained_model.model.predict(self.trainer.X_test)  # type: ignore
        
    def set_prediction_on_train(self):
        """
        Generate and set the predictions on training data.
        
        Raises:
            ValueError: If the training data is not set in the trainer instance.
        """
        # Check if training data is available.
        if self.trainer.X_train is None:
            raise ValueError("Training data not set!")
        
        # Generate predictions on training data and store it.
        self.prediction_on_train = self.trainer.trained_model.model.predict(self.trainer.X_train)  # type: ignore

    def generate_prediction_on_custom_data(self, custom_data: Any):
        """
        Generate predictions on custom data.
        
        Parameters:
            custom_data (Any): The custom data on which to generate predictions.
        
        Returns:
            Any: The generated predictions on the custom data.
        """
        # Generate and return predictions on custom data.
        return self.trainer.trained_model.model.predict(custom_data)  # type: ignore
