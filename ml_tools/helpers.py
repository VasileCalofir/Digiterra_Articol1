from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import copy

def create_simulation(base_simulation, features, model, option, models_config):
    current_simulation = copy.deepcopy(base_simulation)
    current_simulation["features"] = features
    current_simulation["model"]["type"] = model
    
    if option == "with_hyper_params_optimisation":
        hyper_params = models_config.get_hyper_params(model, option)
        current_simulation["model"]["hyper_parameters"] = hyper_params
    else:
        current_simulation["model"]["hyper_parameters"] = {}
        
    return current_simulation


def get_processed_data_from_csv(file_path: str):

    df = pd.read_csv(file_path) 

    return df

def get_pareto_front(pareto_length : int, x : List[float], y : List[float]) -> Tuple[List[float], List[float]]:
    """Gets the pareto front from a list of pareto values.

    Args:
        pareto_length (int): max desired length for the pareto front
        x (List[float]): the x coordinates of the pareto points
        y (List[float]): the y coordinates of the pareto points

    Returns:
        Tuple[List[float], List[float]]: first list represents the x coordinates of the Pareto front, second list represents the y coordinates of the front
    """
    
    pareto_points : List[Tuple[float, float]] = []
    for i in range(len(x)):
        if x[i] >= 0: # if the value is positive
            pareto_points.append((x[i], y[i])) # group the coordinates in a tuple

    dominant_values = []
    i = 0
    pareto_points = sorted(pareto_points, key=lambda tup: tup[0]) # sort the pareto points based on the x values

    while i < len(pareto_points) and len(dominant_values) < pareto_length:
        current_element = pareto_points[i] # get the current element
        dominant_values.append(current_element) # assign it to the dominant values list
        for j in range(i + 1, len(pareto_points)): 
            next_element = pareto_points[j] # get the next element
            if current_element[1] > next_element[1]: # if the current element is dominant
                i = j # repeat the iteration starting from the next element
                break

    sorted_data = sorted(dominant_values, key=lambda tup: tup[1]) # sort the dominant values based on y
    return [x[0] for x in sorted_data] , [x[1] for x in sorted_data]

