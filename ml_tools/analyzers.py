import csv
import json
import os
from typing import List
from sklearn import metrics
from config import Config
from ml_tools.models import Model
from ml_tools.predictor import Predictor
import numpy as np

from ml_tools.trainer import Trainer

# Mapping of performance indicators to their respective sklearn metric functions
INDICATOR_FUNCTION_MAP = {"MAE": metrics.mean_absolute_error,
                          "MSE": metrics.mean_squared_error,
                          "R2": metrics.r2_score}

class Performance:
    """
    Represents a performance metric for a trained machine learning model.
    
    Attributes:
        value (float): The performance metric value.
        indicator (str): The type of performance indicator (e.g., "MAE").
        model (Model): The trained machine learning model.
        data_type (str): The type of data used for evaluation ("train" or "test").
    """
    def __init__(self, simulation: str, value: float, model: Model, indicator: str, data_type: str):
        """Initialize the Performance object."""
        self.simulation = simulation
        self.value: float = value
        self.indicator: str = indicator
        self.model: Model = model
        self.data_type: str = data_type
    
    def __str__(self) -> str:
        """String representation of the Performance object."""
        return f"{self.indicator} ({self.data_type}): {self.value}"

class Reader:
    """
    Reads performance metrics of a machine learning model on training and testing data.
    
    Attributes:
        trainer (Trainer): An object of the Trainer class containing the trained model.
        predictor (Predictor): An object of the Predictor class containing predictions.
        performances (List[Performance]): A list of performance metrics.
    """
    def __init__(self, simulation: str, trainer: Trainer, predictor: Predictor):
        """Initialize the Reader object."""
        self.simulation = simulation
        # Check if the model in trainer is fitted
        if trainer.trained_model is None:
            raise ValueError("Estimation model is not fitted!")
        
        # Check if predictions on test data are available
        if predictor.prediction_on_test is None:
            raise ValueError("Prediction on test not done!")
        
        self.trainer: Trainer = trainer
        self.predictor: Predictor = predictor
        self.performances: List[Performance] = []
        
    def set_performances_on_test_data(self):
        """Calculate and set performance metrics on test data."""
        
        # Check if predictions on test data are available
        if self.predictor.prediction_on_test is None:
            raise ValueError("Prediction on test data not done!")
        
        # Calculate performance metrics for test data
        for indicator, function in INDICATOR_FUNCTION_MAP.items():
            test_perf = Performance(self.simulation,
                                    round(function(self.trainer.y_test, self.predictor.prediction_on_test), 2),
                                    self.trainer.trained_model,  # type: ignore
                                    indicator,
                                    data_type="test")
            self.performances += [test_perf]

    def set_performances_on_train_data(self):
        """Calculate and set performance metrics on training data."""
        
        # Check if predictions on train data are available
        if self.predictor.prediction_on_train is None:
            raise ValueError("Prediction on train data not done!")
        
        # Calculate performance metrics for train data
        for indicator, function in INDICATOR_FUNCTION_MAP.items():
            train_perf = Performance(self.simulation,
                                     round(function(self.trainer.y_train, self.predictor.prediction_on_train), 2),
                                     self.trainer.trained_model,  # type: ignore
                                     indicator,
                                     data_type="train")
            self.performances += [train_perf]
    
    def print_performances(self):
        """Print all performance metrics."""
        
        output = f"{self.trainer.trained_model}: "
        for performance in self.performances:
            output = output + str(performance) + " | "
            
        print(output)
        
class Writer:
    
    def __init__(self, config: Config, simulation: str, performances: List[Performance], results_file_name: str):
        """Writer."""
        
        self.performances = performances
        self.results_file_name = results_file_name
        self.data_for_writing : List[List[str]] = list()
        self.header = ["Simulation", "Description","Features", "Hyper-params", 'Model', 'Data type','Indicator', 'Value']
        self.simulation_name = simulation
        self.config = config
        self.prepare_data_for_writing()
        
    def prepare_data_for_writing(self):
        
        for performance in self.performances:
            performance_data = [performance.model, 
                                performance.data_type, 
                                performance.indicator, 
                                performance.value]
            description = self.config.get_description(self.simulation_name)
            features = ", ".join(self.config.get_features(self.simulation_name))
            hyper_params = json.dumps(self.config.get_hyper_parameters(self.simulation_name))
            row = [self.simulation_name, description, features, hyper_params] + performance_data
            self.data_for_writing.append(row)
    
    def append_to_csv(self):

        file_path = f'{self.results_file_name}.csv'
        
        # Check if the file exists
        file_exists = os.path.exists(file_path)
        
        # Open the file in append mode ('a')
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)

            # If the file did not exist, write the header
            if not file_exists and self.header:
                writer.writerow(self.header)

            # Write the data rows
            for row in self.data_for_writing:
                writer.writerow(row)
