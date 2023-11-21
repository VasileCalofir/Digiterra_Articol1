import json

class Config:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as f:
            self.config_data = json.load(f)

    def get_simulations(self):
        return self.config_data.get('simulations', {})

    def get_simulation(self, simulation_name):
        simulations = self.get_simulations()
        return simulations.get(simulation_name, {})

    def get_description(self, simulation_name):
        simulation = self.get_simulation(simulation_name)
        return simulation.get('description', '')

    def get_features(self, simulation_name):
        simulation = self.get_simulation(simulation_name)
        return simulation.get('features', [])

    def get_model_type(self, simulation_name):
        simulation = self.get_simulation(simulation_name)
        model = simulation.get('model', {})
        return model.get('type', '')

    def get_hyper_parameters(self, simulation_name):
        simulation = self.get_simulation(simulation_name)
        model = simulation.get('model', {})
        return model.get('hyper_parameters', {})


class ModelsConfig:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as f:
            self.config_data = json.load(f)
    
    def get_model_names(self):
        return list(self.config_data['models'].keys())

    def get_hyper_params(self, model_name, param_type='typical_hyper_params'):
        return self.config_data['models'].get(model_name, {}).get(param_type, {})

    def get_all_hyper_params(self, model_name):
        return {
            'no_hyper_params': self.get_hyper_params(model_name, 'no_hyper_params'),
            'typical_hyper_params': self.get_hyper_params(model_name, 'typical_hyper_params'),
            'best_hyper_params': self.get_hyper_params(model_name, 'best_hyper_params')
        }
    
    def get_pareto_color(self, model, hyper_param_option):
        if hyper_param_option:
            return self.config_data['models'][model]["pareto_color_with_hyper_params"]
        else:
            return self.config_data['models'][model]["pareto_color_no_hyper_params"]



class FeatureConfigReader:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = self._read_json_file()

    def _read_json_file(self):
        with open(self.json_file_path, 'r') as f:
            return json.load(f)
    
    def get_features(self, feature_set_name):
        return self.data.get(feature_set_name, {}).get('set', [])

    def get_pareto_marker(self, feature_set_name):
        return self.data.get(feature_set_name, {}).get('pareto_marker', None)
    
    def get_all_feature_set_names(self):
        return list(self.data.keys())

    def get_feature_set_data(self, feature_set_name):
        feature_set_data = self.data.get(feature_set_name, {})
        return feature_set_data.get('set', []), feature_set_data.get('pareto_marker', None)
    
    def get_all_features_dictionary(self):
        feature_dict = {}
        for feature_set_name in self.get_all_feature_set_names():
            feature_dict[feature_set_name] = self.get_features(feature_set_name)
        return feature_dict


class HistogramIntervalsConfigReader:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = self._read_json_file()
        
    def _read_json_file(self):
        with open(self.json_file_path, 'r') as f:
            return json.load(f)
        
    def get_list_of_intervals_for_param(self, param):
        return self.data[param]

# Example usage
if __name__ == '__main__':
    config = Config('path/to/your/config.json')

    print("Simulations:", config.get_simulations())
    print("Features for 'simulation 1':", config.get_features('simulation 1'))
    print("Model type for 'simulation 1':", config.get_model_type('simulation 1'))
    print("Hyperparameters for 'simulation 1':", config.get_hyper_parameters('simulation 1'))
    
    
    # Models config Example usage:

    config = ModelsConfig('path/to/your/config.json')  # Replace with the actual path to your JSON config file

    # Get all model names
    print("All Model Names:", config.get_model_names())

    # Get typical hyperparameters for ridge regression
    print("Typical Hyperparameters for Ridge Regression:", config.get_hyper_params('ridge regression'))

    # Get all hyperparameters for MLP
    print("All Hyperparameters for MLP:", config.get_all_hyper_params('mlp'))
    
