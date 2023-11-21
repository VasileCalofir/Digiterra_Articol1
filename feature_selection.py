import json
import copy
import os
from typing import List
from sklearn.model_selection import train_test_split
from ml_tools.analyzers import Reader, Writer
from ml_tools.plots import FeatureImportancePlotter
from ml_tools.predictor import Predictor
from ml_tools.helpers import create_simulation, get_processed_data_from_csv
from ml_tools.trainer import Trainer
from config import FeatureConfigReader, ModelsConfig, Config

# File paths
feature_selection_config_file_name = os.path.join("config", "feature_selection.json")
features_config_file_name = os.path.join("config", "features.json")
models_config_file_name = os.path.join("config", "models.json")
data_file_name = os.path.join("CSV_files", "data_P3.csv")
results_file_name = "results_feature_selection"

# Configuration stage
models_config = ModelsConfig(models_config_file_name)
models = models_config.get_model_names()

#features_config = FeatureConfig(features_config_file_name)
#features = features_config.get_all_features()

features_config = FeatureConfigReader(features_config_file_name)

features = features_config.get_features("all_features")

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
        current_simulation = create_simulation(base_simulation, [feature], model, "no_hyper_params", models_config )
        
        all_simulations["simulations"][f"simulation {simulation_count}"] = current_simulation
        
        simulation_count += 1

with open(feature_selection_config_file_name, "w") as f:
    json.dump(all_simulations, f, indent=4)
    
# ML Analysis  
processed_data = get_processed_data_from_csv(data_file_name)

y = processed_data['DI_cladire'] 

config = Config(feature_selection_config_file_name)

simulations = config.get_simulations()

readers: List[Reader] = []
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
    readers.append(reader)
    reader.set_performances_on_test_data()
    reader.set_performances_on_train_data()

    print("---------------------------")
    reader.print_performances()
    writer = Writer(config, simulation,reader.performances, results_file_name)
    writer.append_to_csv()



indicators = ["R2", "MAE"]
models = [reader.trainer.model_type for reader in readers]

performances_by_indicator_by_model = {}
for indicator in indicators:
    performances_by_indicator_by_model[indicator] = {model: [] for model in models  }

for reader in readers:
    model = reader.trainer.model_type
    for indicator in indicators:
        performance =[performance.value for performance in reader.performances if performance.indicator == indicator and performance.data_type == "test" and reader.trainer.model_type == str(performance.model)][0]
        performances_by_indicator_by_model[indicator][model].append(performance)

for model in models:
    for indicator in indicators:
        plotter = FeatureImportancePlotter(config,"figs", True, features, performances_by_indicator_by_model[indicator][model], indicator, model)
        plotter.generate_plot()