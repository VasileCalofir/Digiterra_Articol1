from typing import Dict, List, Tuple, Union
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Custom modules for data scaling and model selection
from Data_Partea3 import MinMaxScaling, PCA_fs, RobustScaling, SelectKBest_Anova, SelectKBest_cS, StandardScaling, UnitVectorScaling
from ml_tools.models import KNN, MLP, GradientBoostingModel, LassoRegressionModel, LinearRegressionModel, RandomForrestModel, RidgeRegressionModel, SVRModel

class Trainer:
    """
    A trainer class designed to facilitate the training of prediction models.
    
    Attributes:
        X_train (pd.DataFrame): The feature matrix used for training.
        y_train (pd.Series): The target vector used for training.
        X_test (pd.DataFrame): The feature matrix used for testing.
        y_test (pd.Series): The target vector used for testing.
        trained_model: The trained machine learning model.
        orig_x (pd.DataFrame): Original feature matrix.
        orig_y (pd.Series): Original target vector.
        model_type (str): Type of model to train (e.g., "linear_regression").
        
    """

    def __init__(self, orig_x: pd.DataFrame, orig_y: pd.Series, model_type="linear_regression", hyper_params: Dict=dict(), use_min_max_scale: bool=False):
        """
        Initializes the Trainer class with original feature matrix and target vector.
        
        Parameters:
            orig_x (pd.DataFrame): Original feature matrix.
            orig_y (pd.Series): Original target vector.
            model_type (str): Type of model to train (default is "linear_regression").
        """
        self.X_train: pd.DataFrame = pd.DataFrame({})  # Initialize training feature matrix
        self.y_train: pd.Series = pd.Series(float)  # Initialize training target vector
        self.X_test: pd.DataFrame = pd.DataFrame({})  # Initialize testing feature matrix
        self.y_test: pd.Series = pd.Series(float)  # Initialize testing target vector
        self.trained_model = None  # Initialize trained model
        
        if use_min_max_scale:
            self.X = self.scale_data(orig_x) #type: ignore
        else:
            self.X: pd.DataFrame = orig_x 
            
        self.y: pd.Series = orig_y  
        self.model_type = model_type  
        self.hyper_params = hyper_params

    def scale_data(self, x_data: pd.DataFrame):
        scaler = MinMaxScaler()
        scaler.fit(x_data)
        scaled_data = scaler.transform(x_data)
        return scaled_data
    
    def set_train_test_data(self, test_size: float = 0.2):
        """
        Splits the original data into training and test sets.
        
        Parameters:
            test_size (float): Proportion of the data to be used for testing (default is 0.2).
        """
        # Split the data into training and testing sets. 
        # Here, test_size=0.2 means 20% of the data is used for testing, and 80% for training.
       
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42)

    def train_model(self):
        """
        Trains a machine learning model based on the selected model type.
        """
        
        if self.model_type == "gradient boosting":
            self.trained_model = GradientBoostingModel(self.X_train, self.y_train, self.hyper_params)

        if self.model_type == "svr":
            self.trained_model = SVRModel(self.X_train, self.y_train, self.hyper_params)

        if self.model_type == "linear regression":
            self.trained_model = LinearRegressionModel(self.X_train, self.y_train, self.hyper_params)

        if self.model_type == "random forrest":
            self.trained_model = RandomForrestModel(self.X_train, self.y_train, self.hyper_params)
            
        if self.model_type == "lasso regression":
            self.trained_model = LassoRegressionModel(self.X_train, self.y_train, self.hyper_params)
            
        if self.model_type == "ridge regression":
            self.trained_model = RidgeRegressionModel(self.X_train, self.y_train, self.hyper_params)
            
        if self.model_type == "knn":
            self.trained_model = KNN(self.X_train, self.y_train, self.hyper_params)
        
        if self.model_type == "mlp":
            self.trained_model = MLP(self.X_train, self.y_train, self.hyper_params)


