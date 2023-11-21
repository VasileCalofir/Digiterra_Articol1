
from abc import ABC, abstractmethod
import os
from textwrap import wrap
from typing import List

import pandas as pd
from config import Config, HistogramIntervalsConfigReader
from ml_tools.helpers import get_pareto_front
from ml_tools.predictor import Predictor
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from ml_tools.trainer import Trainer


class Plotter(ABC):
    
    def __init__(self, folder_path: str, save_figs: bool):
        
        self.save_figs = save_figs
        if save_figs:
            # os.environ['PATH'] = os.environ['PATH'] + ':/Users/mirceastefansimoiu/bin'  # For Mac os, add Latex to plots
            os.environ['PATH'] = os.environ['PATH'] + ':/usr/bin/pdflatex'  # Add LaTeX path for Ubuntu
            plt.rc('text', usetex=True) # add latex to plot if the user wants to save it
            plt.style.use(['science'])
            plt.grid(True)
        
        self.folder_path = folder_path
        self.title = ""
        self.plot_type = ""
        self.data_type = ""
    
    def create_figure(self):
        plt.figure(figsize=(8, 6))
    
    def set_x_label(self, text):
        self.x_label = text
    
    def set_y_label(self, text):
        self.y_label = text
        
    def add_plot_elements(self):
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
    
    def save_fig(self):
        file_path = os.path.join(self.folder_path, f"Figure_{self.plot_type}_{self.data_type}.pdf")
        plt.savefig(file_path)
        plt.close()
        
    def output_figure(self):
        if self.save_figs:
            self.save_fig()
        else:
            plt.show()
     
    @abstractmethod   
    def plot(self):
        pass
    
    @abstractmethod   
    def generate_plot(self):
        pass

class SimulationPlotter(Plotter, ABC):
    
    def __init__(self, config: Config, folder_path: str, save_figs: bool):
        super().__init__(folder_path, save_figs)

        self.config = config
     
    
class GlobalPlotter(Plotter, ABC):
    
    def __init__(self, folder_path: str, save_figs: bool):
        super().__init__(folder_path, save_figs)

    
    
class RealPredictedPlotter(SimulationPlotter):
    
    def __init__(self, config: Config, folder_path: str, save_figs: bool, simulation: str, feature_set:str, features: List, trainer: Trainer, predictor: Predictor):
        super().__init__(config, folder_path, save_figs)
        
        self.features = ", ".join(features)
        self.simulation = simulation
        self.trainer = trainer
        self.predictor = predictor
        self.plot_type = f"real_vs_predicted_{trainer.trained_model}_{feature_set}"
        self.data_type = "test"
    
    def generate_plot(self):
        self.create_figure()
        self.set_x()
        self.set_x_label('DI Index real values')
        self.set_y()
        self.set_y_label('DI Index predicted values')
        self.set_complete_model_name()
        self.set_title()
        
        self.plot()
        self.add_plot_elements()
        self.output_figure()
    
    def set_complete_model_name(self):
        self.model = [f"optimised {model}" if self.trainer.hyper_params else model for model in [self.trainer.model_type]][0]
        
    def set_x(self):
        if self.data_type == "test":
            self.x = self.trainer.y_test
        else:
            self.x = self.trainer.y_train
    
    def set_y(self):
        self.y = self.predictor.prediction_on_test
        
    
    def set_title(self):
        figure_title = f'Real vs. predicted analysis for the test dataset, using {self.model} and the following features: {self.features} '
        self.title = "\n".join(wrap(figure_title, 60))  # Wrap text at 60 characters
        
    def plot(self):
        
        plt.scatter(self.x, self.y, c='blue')

        plt.plot([min(self.x), max(self.x)], [min(self.x), max(self.x)], c='red', linestyle='--')

          
class FeatureImportancePlotter(SimulationPlotter):
    
    def __init__(self, config: Config, folder_path: str, save_figs: bool, features: List, importances: List, indicator: str, model: str):
        super().__init__(config, folder_path, save_figs)
        
        self.features = features
        self.importances = importances
        self.plot_type = f"feature_importance_{indicator}_{model}"
        self.data_type = "test"
        self.model = model
        self.indicator = indicator
        
    def generate_plot(self):
        self.create_figure()
        self.set_x_label(self.indicator)
        self.set_y_label('Features')
        self.set_title()
        self.set_df_for_plotting()
        
        self.plot()
        self.add_plot_elements()
        self.output_figure()
            
    def set_df_for_plotting(self):

        self.df = pd.DataFrame({'Feature': self.features, 'Importance':self.importances})
        self.df = self.df.sort_values('Importance', ascending=True)
        
    def set_title(self):
        self.title = f'Feature importance, using {self.model}'
            
    def plot(self):
        
        plt.barh(self.df['Feature'], self.df['Importance'], color='skyblue')
        
        
class ParetoPlotter(GlobalPlotter):
    
    def __init__(self, folder_path: str, save_figs: bool, x: List, y:List, indicator_x: str, indicator_y: str, legend_elements: List[str], markers, colors):
        super().__init__(folder_path, save_figs)
        
        self.x = x
        self.y = y
        self.plot_type = f"pareto_{indicator_x}_vs_{indicator_y}"
        self.set_x_label(indicator_x)
        self.set_y_label(indicator_y)
        self.data_type = "test"
        self.legend_elements = legend_elements
        self.markers = markers
        self.colors = colors

    def get_paret_values(self):
        self.x_pareto, self.y_pareto = get_pareto_front(10, self.x, self.y)
        
    def generate_plot(self):
        self.create_figure()
        self.get_paret_values()
        self.set_title()
        
        self.plot()
        self.add_plot_elements()
        
        plt.legend(self.legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        self.output_figure()
            
        
    def set_title(self):
        self.title = f'Mapping of {self.x_label} against {self.y_label}'
            
    def plot(self):
        
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            plt.scatter(x, y, marker=self.markers[i], color = self.colors[i]) 
        
        plt.plot(self.x_pareto, self.y_pareto) 

histo_title = {
    ("Lshape_0","DI_cladire") : "Damage index distribution for regular geometry models",
    ("Lshape_1","DI_cladire") : "Damage index distribution for irregular geometry models",
    ("Lshape_0","no_bay") : "Number of bays (no_bay1) distribution for regular geometry models",
    ("Lshape_1","no_bay") : "Number of bays (no_bay21) distribution for irregular geometry models",
    ("Lshape_1","no_bay_2") : "Number of bays (no_bay22) distribution for irregular geometry models",
    ("Lshape_0","no_span") : "Number of spans (no_span1) distribution for regular geometry models",
    ("Lshape_1","no_span") : "Number of spans (no_span21)  distribution for irregular geometry models",
    ("Lshape_1","no_span_2") : "Number of spans (no_span22) distribution for irregular geometry models",
    ("Lshape_0","no_story") : "Total number of storeys (no_storey1) distribution for regular geometry models",
    ("Lshape_1","no_story") : "Total number of storeys (no_story21) distribution for irregular geometry models",
    ("Lshape_1","no_story_2") : "Total number of storeys (no_story22) distribution for irregular geometry models",
    ("PGA_ALL","PGA_of_the_recording_scale1") : "Distribution of peak ground acceleration for the X component of the seismic input (scaled)",
    ("PGA_ALL","PGA_of_the_recording_scale2") : "Distribution of peak ground acceleration for the Y component of the seismic input (scaled)",
    ("Lshape_0","T1") : "Distribution of the first mode natural period for regular geometry models",
    ("Lshape_1","T1") : "Distribution of the first mode natural period for irregular geometry models",
    }
histo_ox = {
    "DI_cladire" : "Damage index",
    "no_bay" : "Number of bays",
    "no_bay_2" : "Number of bays",
    "no_span" : "Number of spans",
    "no_span_2" : "Number of spans",
    "no_story" : "Number of storeys",
    "no_story_2" : "Number of storeys",
    "PGA_of_the_recording_scale1":"PGA",
    "PGA_of_the_recording_scale2":"PGA",
    "T1":"$1^{st}$ natural period (seconds)"
    }

class HistogramPlotter(GlobalPlotter):
    
    def __init__(self, config: HistogramIntervalsConfigReader,  folder_path: str, save_figs: bool, hist_title: str, data: pd.DataFrame):
        super().__init__(folder_path, save_figs)
        
        # self.config = config
        # self.title_root = file
        # self.plot_type = f"histogram"
        # self.data = data
        # self.set_y_label("")

        self.config = config
        self.title_root = hist_title # denumire csv
        self.plot_type = f"histogram" # din denumurea figurii pdf 
        self.data = data
        self.set_y_label("Number of buildings")
        
    def generate_plot(self):
        
        for column in self.data.columns:
            self.create_figure()
            self.set_title(column)
            self.data_type = f"{self.title_root}_{column}"
            self.set_x_label(histo_ox[column])
            
            self.plot(column)
            
            self.add_plot_elements()
            self.output_figure()
    
    def set_title(self, column_name):
        self.title = histo_title[(self.title_root,column_name)]
    
    def plot(self, column_name):
        if column_name in self.config.data:
            intervals = self.config.get_list_of_intervals_for_param(column_name)
            plt.xticks(intervals)
        else:
            intervals = 20
        
        n, bins, patches = plt.hist(self.data[column_name], bins=intervals, alpha=0.75, edgecolor='black', rwidth=0.9,align='left')

        # Check if 'n' and 'bins' have expected lengths
        if len(n) > 0 and len(bins) == len(n) + 1:
            # Annotate the values on top of the bars.
            for i in range(len(n)):
                # Calculate the center of each bin
                bin_center = bins[i]
                # Annotate the count above each bar
                if int(n[i]) != 0:
                    plt.annotate(str(int(n[i])), xy=(bin_center, n[i]), xytext=(0,5),
                                textcoords="offset points", ha='center', va='bottom')
        
        