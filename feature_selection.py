import json
import copy
import os
from sklearn.model_selection import train_test_split
from ml_tools.analyzers import Reader, Writer
from ml_tools.predictor import Predictor
from ml_tools.helpers import get_processed_data_from_csv
from ml_tools.trainer import Trainer
from config import FeatureConfig, ModelsConfig, Config

# File paths
feature_selection_config_file_name = os.path.join("config", "feature_selection.json")
features_config_file_name = os.path.join("config", "features.json")
models_config_file_name = os.path.join("config", "models.json")
data_file_name = "data_P3.csv"
results_file_name = "results_feature_selection"

# Configuration stage
models_config = ModelsConfig(models_config_file_name)
models = models_config.get_model_names()

features_config = FeatureConfig(features_config_file_name)
features = features_config.get_all_features()

# Define the static portions of your simulation configurations
base_simulation = {
    "description": 'feature selection',
    "features": [],
    "model": {
        "type": "",
        "hyper_parameters": {}
    }
}

# Initialize the dictionary to hold all simulations
all_simulations = {"simulations": {}}

# Iterate over features, models to populate the simulations dictionary
simulation_count = 1
for feature in features:
    for model in models:
        # Create a deep copy of the base_simulation dictionary
        current_simulation = copy.deepcopy(base_simulation)
        
        current_simulation["features"].append(feature)
        current_simulation["model"]["type"] = model
        current_simulation["model"]["hyper_parameters"] = {}
        
        all_simulations["simulations"][f"simulation {simulation_count}"] = current_simulation
        
        simulation_count += 1

with open(feature_selection_config_file_name, "w") as f:
    json.dump(all_simulations, f, indent=4)
    
# ML Analysis  
processed_data = get_processed_data_from_csv(data_file_name)

y = processed_data['DI_cladire'] 

config = Config(feature_selection_config_file_name)

simulations = config.get_simulations()
for simulation in simulations:

    selected_features = config.get_features(simulation)
    x = processed_data[selected_features]
    model_type = config.get_model_type(simulation)
    hyper_params = config.get_hyper_parameters(simulation)
    
    trainer = Trainer(x, y, model_type, hyper_params)
    trainer.set_train_test_data(test_size=0.2)
    trainer.train_model()
    
    predictor = Predictor(trainer)
    predictor.set_prediction_on_test()
    predictor.set_prediction_on_train()

    reader = Reader(simulation, trainer, predictor)
    reader.set_performances_on_test_data()
    reader.set_performances_on_train_data()

    print("---------------------------")
    reader.print_performances()
    writer = Writer(config, simulation,reader.performances, results_file_name)
    writer.append_to_csv()
