
import os
import pandas as pd
from config import HistogramIntervalsConfigReader

from ml_tools.plots import HistogramPlotter

import matplotlib.pyplot as plt

files = ['Lshape_0', 'Lshape_1', 'PGA_ALL']

# Directorul cu fișierele CSV
csv_folder_path = "CSV_files"

config_file_path = os.path.join("config", "histogram_intervals.json")

config = HistogramIntervalsConfigReader(config_file_path)

for file in files:
    # Construiește calea către fișierul CSV din directorul specific
    file_path = os.path.join(csv_folder_path, f"{file}.csv")

    # Verifică dacă fișierul există înainte de a încerca să-l citim
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path)
        plotter = HistogramPlotter(config, "figs", True, file, data)
        plotter.generate_plot()
    else:
        print(f"Fișierul {file} nu există în directorul {csv_folder_path}.") 