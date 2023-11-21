import json

import os
from sklearn.model_selection import train_test_split
from ml_tools.analyzers import Reader, Writer
from ml_tools.plots import ParetoPlotter, RealPredictedPlotter
from ml_tools.predictor import Predictor
from ml_tools.helpers import create_simulation, get_processed_data_from_csv
from ml_tools.trainer import Trainer
from config import FeatureConfigReader, ModelsConfig, Config

def create_simulations_config_file(features_set):
# Define the static portions of your simulation configurations
    base_simulation = {
        "description": features_set,
        "features": [],
        "model": {
            "type": "",
            "hyper_parameters": {}
        }
    }

    all_simulations = {"simulations": {}}
    simulation_count = 1

    for model in models:
        options = ["no_hyper_params"] if "regression" in model else ["no_hyper_params", "with_hyper_params_optimisation"]

        for option in options:
            
            current_simulation  = create_simulation(base_simulation, features, model, option, models_config)
            
            # Add the current_simulation dictionary to the all_simulations dictionary
            all_simulations["simulations"][f"simulation {simulation_count} {features_set}"] = current_simulation
            
            simulation_count += 1

    with open(comparison_config_file_name, "w") as f:
        json.dump(all_simulations, f, indent=4)

def get_value_for_pareto_plot(reader, indicator, data_type):
    for performance in reader.performances:
        if performance.indicator == indicator and performance.data_type == data_type:
            if indicator == "R2":
                return (1-performance.value)
            else:
                return performance.value
    return None
            

if __name__ == '__main__':

    # Define file paths and constants
    comparison_config_file_name = os.path.join("config", "comparison.json")
    features_config_file_name = os.path.join("config", "features.json")
    models_config_file_name = os.path.join("config", "models.json")
    data_file_name = os.path.join("CSV_files", "data_P3.csv")
    results_file_name = "results_comparison"
    scaler_options = ["min-max scale"]
    ALL_FEATURES_COUNT = 23

    # Initialize objects for managing configurations and features
    models_config = ModelsConfig(models_config_file_name)
    models = models_config.get_model_names()
    features_config = FeatureConfigReader(features_config_file_name)
    best_features_by_set = features_config.get_all_features_dictionary()

    # Data storage dictionaries and lists
    pareto_data_sets = { 'R2 vs. MAE': { 'R2' : [], 'MAE' : [] },
                         'MAE vs. FUR': { 'MAE' : [], 'FUR' : [] },
                         'R2 vs. FUR': { 'R2' : [], 'FUR' : [] }}

    legend_elements = []
    markers = []
    colors = []
    readers = []
    
    # Main loop through feature sets
    for features_set, features in best_features_by_set.items():
    
        create_simulations_config_file(features_set)

        # Machine learning analysis
        processed_data = get_processed_data_from_csv(data_file_name)

        y = processed_data['DI_cladire'] 

        config = Config(comparison_config_file_name)
        
        simulations = config.get_simulations()
        
        for simulation in simulations:

            selected_features = config.get_features(simulation)
            x = processed_data[selected_features]
            model_type = config.get_model_type(simulation)
            hyper_params = config.get_hyper_parameters(simulation)
            
            if hyper_params:
                option = "with hyper-params"
            else:
                option = ""
            
            if "min-max scale" in scaler_options:
                trainer = Trainer(x, y, model_type, hyper_params, use_min_max_scale=True)
            else:
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
            readers.append(reader)
            
            writer = Writer(config, simulation,reader.performances, results_file_name)
            writer.append_to_csv()
            
            # Collecting performance data for plotting
            pareto_data_sets["R2 vs. MAE"]["R2"].append(get_value_for_pareto_plot(reader, "R2", "test"))
            pareto_data_sets["R2 vs. MAE"]["MAE"].append(get_value_for_pareto_plot(reader, "MAE", "test"))
            pareto_data_sets["R2 vs. FUR"]["R2"].append(get_value_for_pareto_plot(reader, "R2", "test"))
            pareto_data_sets["R2 vs. FUR"]["FUR"].append(len(features)/ALL_FEATURES_COUNT)
            pareto_data_sets["MAE vs. FUR"]["MAE"].append(get_value_for_pareto_plot(reader, "MAE", "test"))
            pareto_data_sets["MAE vs. FUR"]["FUR"].append(len(features)/ALL_FEATURES_COUNT)
            
            legend_elements.append(f"{model_type} {option}, {features_set}")
            
            marker = features_config.get_pareto_marker(features_set)
            markers.append(marker)  
            
            color = models_config.get_pareto_color(model_type, option)
            colors.append(color) 
            
            plotter = RealPredictedPlotter(config, "figs", True, simulation, features_set, features, trainer, predictor)
            plotter.generate_plot()
    
    # Generate Pareto plots
    # Choose False to show figures instead of saving them
    plotter = ParetoPlotter("figs", True, pareto_data_sets["R2 vs. MAE"]["R2"], 
                            pareto_data_sets["R2 vs. MAE"]["MAE"], 
                            "1 - R2", "MAE", legend_elements, markers, colors)
    plotter.generate_plot()    
    
    plotter = ParetoPlotter("figs", True, pareto_data_sets["R2 vs. FUR"]["R2"], 
                            pareto_data_sets["R2 vs. FUR"]["FUR"], 
                            "1 - R2", "FUR", legend_elements, markers, colors)
    plotter.generate_plot()    
    
    plotter = ParetoPlotter("figs", True, pareto_data_sets["MAE vs. FUR"]["MAE"], 
                            pareto_data_sets["MAE vs. FUR"]["FUR"], 
                            "MAE", "FUR", legend_elements, markers, colors)
    plotter.generate_plot()    
           