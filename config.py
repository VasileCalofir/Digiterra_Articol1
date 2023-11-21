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


class FeatureConfig:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as f:
            self.config_data = json.load(f)
    
    def get_all_features(self):
        return self.config_data.get('all_features', [])
    
    def get_best_features(self, method):
        return self.config_data.get(f'best_features_{method}', [])
    
    def get_all_best_features(self):
        return {
            'r2': self.get_best_features('r2'),
            'gradient_boosting': self.get_best_features('gradient_boosting'),
            'random_forrest': self.get_best_features('random_forrest')
        }




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
    
    
    # Feature config Example usage:

    feature_config = FeatureConfig('path/to/your/feature_config.json')  # Replace with the actual path to your JSON config file

    # Get all features
    print("All Features:", feature_config.get_all_features())

    # Get best features as determined by r2 method
    print("Best Features by R2 Method:", feature_config.get_best_features('r2'))

    # Get best features as determined by all methods
    print("Best Features by All Methods:", feature_config.get_all_best_features())