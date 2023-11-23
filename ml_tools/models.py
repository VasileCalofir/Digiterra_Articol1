
from typing import Dict
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR

class Model:
    """
    Base class for different regression models.
    """
    
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict = dict()):
        """
        Initialize the Model class.
        
        Parameters:
            x_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        
        # Raise an error if training data is not set
        if x_train is None or y_train is None:
            raise ValueError("Training data not set!")
        
        self.x_train : pd.DataFrame = x_train
        self.y_train : pd.Series = y_train
        self.model = None
        self.hyper_params = hyperparams
    
    def train_model(self):
        """
        Train the model.
        """
        
        # Raise an error if the model is not initialized
        if self.model is None:
            raise ValueError("Model not set!")
        
        if self.hyper_params:
            grid = GridSearchCV(self.model, self.hyper_params)
            grid.fit(self.x_train, self.y_train)
            self.model = grid.best_estimator_
        else:
            self.model.fit(self.x_train, self.y_train)
        
        
class LinearRegressionModel(Model):
    """
    Class for Linear Regression Model.
    """
    
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict=dict()):
        """
        Initialize the LinearRegressionModel class and train the model.
        
        Parameters:
            x_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        super().__init__(x_train, y_train, hyperparams)
        
        # Initialize the model with hyperparameters
        self.model = LinearRegression(
            fit_intercept=True,    
            copy_X=True,           
            n_jobs=None,            
            positive=False
            )
        
        self.train_model()
        
    def __str__(self) -> str:
        """Returns the name of the model."""
        if self.hyper_params:
            return "Optimised Linear regression"
        else:
            return "Linear regression"
    
class LassoRegressionModel(Model):
    """
    Class for Lasso Regression Model.
    """
    
    def __init__(self,  x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict=dict()):
        super().__init__(x_train, y_train, hyperparams)
        
        # Initialize the model
        self.model = Lasso(
            fit_intercept=True,    # Specifică dacă se dorește includerea termenului de interceptare în model. Valorile True sau False.
            copy_X=True,           # Specifică dacă să se copieze sau să se suprascrie variabila de intrare X. Valorile True sau False.          # Specifică numărul de job-uri paralele pentru a fi utilizate în timpul ajustării modelelor. Default este None.
            positive=False
            )
        
        self.train_model()
        
    def __str__(self) -> str:
        """Returns the name of the model."""
        if self.hyper_params:
            return "Optimised Lasso regression"
        else:
            return "Lasso regression"
    
class RidgeRegressionModel(Model):
    
    def __init__(self,  x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict=dict()):
        super().__init__(x_train, y_train, hyperparams)
        
        # Initialize the model
        self.model = Ridge(
            fit_intercept=True,    # Specifică dacă se dorește includerea termenului de interceptare în model. Valorile True sau False.
            copy_X=True,           # Specifică dacă să se copieze sau să se suprascrie variabila de intrare X. Valorile True sau False.          # Specifică numărul de job-uri paralele pentru a fi utilizate în timpul ajustării modelelor. Default este None.
            positive=False
            )
        
        self.train_model()
        
    def __str__(self) -> str:
        if self.hyper_params:
            return "Optimised Ridge regression"
        else:
            return "Ridge regression"
    
class SVRModel(Model):
    
    def __init__(self,  x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict=dict()):
        super().__init__(x_train, y_train, hyperparams)
        
        # Initialize the model
        self.model = LinearSVR()
        
        self.train_model()
        
    def __str__(self) -> str:
        if self.hyper_params:
            return "SV regression"
        else:
            return "Optimised SV regression"
    
class GradientBoostingModel(Model):
    
    def __init__(self,  x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict=dict(), print_feature_importance: bool = False):
        super().__init__(x_train, y_train, hyperparams)
        
        # Initialize the model
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        
        self.train_model()
        
        if print_feature_importance:
            # Get feature importances
            feature_importances = self.model.feature_importances_
            
            # Print or store feature importances
            feature_names = self.model.feature_names_in_
            feature_importance_df = pd.DataFrame({
                'Feature Name': feature_names,
                'Importance': feature_importances
            })
            print(feature_importance_df.sort_values(by='Importance', ascending=False))
        
    def __str__(self) -> str:
        if self.hyper_params:
            return "Optimised Gradient boosting"
        else:
            return "Gradient boosting"
         
        
        
class RandomForrestModel(Model):
    
    def __init__(self,  x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict=dict(), print_feature_importance: bool = False):
        super().__init__(x_train, y_train, hyperparams)
        
        # Initialize the model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.train_model()
        
        if print_feature_importance:
            # Get feature importances
            feature_importances = self.model.feature_importances_
            
            # Print or store feature importances
            feature_names = self.model.feature_names_in_
            feature_importance_df = pd.DataFrame({
                'Feature Name': feature_names,
                'Importance': feature_importances
            })
            print(feature_importance_df.sort_values(by='Importance', ascending=False))
        
    def __str__(self) -> str:
        if self.hyper_params:
            return "Optimised Random forrest"
        else:
            return "Random forrest"
        
class KNN(Model):
    
    def __init__(self,  x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict=dict()):
        super().__init__(x_train, y_train, hyperparams)
        
        self.model = KNeighborsRegressor(n_neighbors=5)
        
        self.train_model()
        
    def __str__(self) -> str:
        if self.hyper_params:
            return "Optimised K Nearest Neighbours"
        else:
            return "K Nearest Neighbours"
    
class MLP(Model):
    
    def __init__(self,  x_train: pd.DataFrame, y_train: pd.Series, hyperparams: Dict=dict()):
        super().__init__(x_train, y_train, hyperparams)
        
        #self.model = MLPRegressor(hidden_layer_sizes=(1, 1), max_iter=1000)
        self.model = MLPRegressor(max_iter=1000)
        
        self.train_model()
        
    def __str__(self) -> str:
        if self.hyper_params:
            return "Optimised MLP"
        else:
            return "MLP"
    
