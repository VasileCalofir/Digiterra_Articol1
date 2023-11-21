import pandas as pd
from sklearn.model_selection import train_test_split



def get_processed_data_from_csv(file_path: str):

    df = pd.read_csv(file_path) 

    return df

