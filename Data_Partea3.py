import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,Normalizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.decomposition import PCA
import re
import os


def Simulation_Data_Cleaning(path_sim):
    # Numele coloanelor
    columnNames=['Recording_index','Timestep','Scale_factor','PGA_of_the_recording1','PGA_of_the_recording2','Direction','RX_max',
              'UXMAX','UYMAX','Hieght','b_Hieght','d_cmcr','span','bay','no_story','no_span','no_bay','csi','b_st','h_st','b_gr','h_gr','Np','Shift',
                'py','ro','E','MstY','MstX','Mgr','Diaphragm','Lshape','bay2','no_span_2','no_bay_2','no_story_2','lim','T1','T2','T3',
                'nr_DOF','ne','nn','nph-number_of_plastic_hings','assembling_time','solver_time','Damage_eval_dur','driftmax_X','driftmax_Y',
                'FbXmax','FbYmax','nr_ph_max','nr_ph_max/nph','HE(i)/IE(i)','DI_cladire']

    # Get a list of all CSV files in the folder
    data_sim = glob.glob(path_sim + "/*.csv")

    # Create an empty list to store the data frames
    list_of_dfs = []

    # Loop through each CSV file and append its data frame to the list
    for filename in data_sim:
        df = pd.read_csv(filename, header=None) # Specify header=None to indicate that the CSV file doesn't have a header row
        list_of_dfs.append(df)

    # Concatenate all the data frames in the list into a single data frame
    combined_df = pd.concat(list_of_dfs, ignore_index=True)

    # delete rows with NaN values
    df_data_sim = combined_df.dropna()

    # Set the column names
    df_data_sim.columns = columnNames

    
    # Delete rows where 'DI_cladire' is less than 0 or greater than 2.5
    df_data_sim = df_data_sim[(df_data_sim['DI_cladire'] >= 0) & (df_data_sim['DI_cladire'] <= 2.5)]
    
    return df_data_sim


def Data_Set_Earthquakes(folder_path):
    # Create an empty DataFrame with column names
    df_acc = pd.DataFrame(columns=['Recording_index','AccName1','Fund_Freq1','AbsoluteAmplitude1','AccName2','Fund_Freq2','AbsoluteAmplitude2'])
    i = 1

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, file)
            
               # Get the list of column names
            df = pd.read_csv(csv_file_path)
            column_names = df.columns.tolist()

            # Separate the column names into two variables
            column_name_1 = column_names[0]
            column_name_2 = column_names[1]

            # Initialize numar_csv with None before attempting to assign it
            numar_csv = None

            # Verify if there is at least one digit in the file name
            if re.match(r'^\d+\.csv$', file):
                # Verify if the file name matches the pattern of a string of digits followed by ".csv"
                nume_fisier_fara_extensie = os.path.splitext(file)[0]
                numar_csv = int(nume_fisier_fara_extensie)
                #print(numar_csv)
            else:
                print(f"Error: No valid number found in the file name '{file}'.")

            signal1 = df.iloc[:, 0]
            fund_freq1 = Find_fft(signal1)

            signal2 = df.iloc[:, 1]
            fund_freq2 = Find_fft(signal2)

            # Find the absolute maximum of an accelerogram
            abs_max1 = max(signal1, key=abs)
            abs_max2 = max(signal2, key=abs)

            # Set values in specific cells
            df_acc.at[i, 'AccName1'] = column_name_1
            df_acc.at[i, 'AccName2'] = column_name_2
            df_acc.at[i, 'Recording_index'] = numar_csv
            df_acc.at[i, 'Fund_Freq1'] = np.abs(fund_freq1)
            df_acc.at[i, 'AbsoluteAmplitude1'] = abs_max1
            df_acc.at[i, 'Fund_Freq2'] = np.abs(fund_freq2)
            df_acc.at[i, 'AbsoluteAmplitude2'] = abs_max2
            # print(i)
            
            # Verificați dacă 'Fund_Freq1' sau 'Fund_Freq2' are valoarea 0
            if fund_freq1 == 0 or fund_freq2 == 0:
                print(f"Row {i-1}:")
                print(df_acc.iloc[i-1])
                print("\n")
            i+=1
    return df_acc


def Data_Partea3(sel):

    ###### CREATE A DATA SET WITH BULTINGS AND EARTHQUAQUE MOST RELEVANT FEATURES ################
    
    allEarthqParam = pd.read_csv(r'AccSet_P3_Clean.csv')
    allSimRez = pd.read_csv(r'DataSet_P3.csv')    

    # selectedFeature = ['Recording_index', 'Scale_factor', 'Hieght', 'Span','bay', 'no_story', 'no_span', 'no_bay', 'csi',
    #                    'b_st', 'h_st', 'b_gr', 'h_gr', 'Np', 'Shift', 'py', 'ro', 'E', 'MstY', 'MstX', 'Mgr', 'Diaphragm',
    #                    'Lshape', 'bay2', 'no_span_2', 'no_bay_2', 'no_story_2', 'T1', 'T2', 'T3', 'DI_cladire']

    # actualizeaza PGA-ul multiplicat cu factorul de scalare
    allSimRez['PGA_of_the_recording_scale_1'] = allSimRez['Scale_factor'] * allSimRez['PGA_of_the_recording1']
    allSimRez['PGA_of_the_recording_scale_2'] = allSimRez['Scale_factor'] * allSimRez['PGA_of_the_recording2']
    # creeaza 2 parametri noi combinati
    allSimRez['dim_x'] = allSimRez['span'] * allSimRez['no_span']
    allSimRez['dim_y'] = allSimRez['bay'] * allSimRez['no_bay']   
    selectedFeature=['Recording_index','PGA_of_the_recording_scale_1','PGA_of_the_recording_scale_2','b_Hieght','dim_x',
                     'dim_y','b_st','h_st','b_gr','h_gr','E','MstY','MstX','Mgr','Lshape','bay2',
                     'no_span_2','no_bay_2','no_story_2','T1','T2','T3','DI_cladire']
    allSimRez=allSimRez[selectedFeature]
    allSimRez['Recording_index'] = allSimRez['Recording_index'].astype('int')

    # create sample dataframes with a common index
    df1 = allSimRez
    df2 = allEarthqParam

    # merge the two dataframes on the common index
    merged_df = pd.merge(df1, df2, on='Recording_index', how='left')
    
    ####### RENAME FEATURES IN A MORE ADEQUATE WAY ################## 

    newSelectedFeature=['Fund_Freq1','Fund_Freq2','PGA_of_the_recording_scale_1','PGA_of_the_recording_scale_2','b_Hieght','dim_x',
                     'dim_y','b_st','h_st','b_gr','h_gr','E','MstY','MstX','Mgr','Lshape','bay2',
                     'no_span_2','no_bay_2','no_story_2','T1','T2','T3','DI_cladire']
    merged_df = merged_df[newSelectedFeature]
    
    # CONVERT FEATURES IN A NUMERICAL FORMAT#############  
    merged_df['Fund_Freq1'] = merged_df['Fund_Freq1'].astype(float)
    merged_df['Fund_Freq2'] = merged_df['Fund_Freq2'].astype(float)
    
    ############ ROUND FEATURES AT 3 DECIMALS###############
    # merged_df[['Fund_Freq','T1', 'T2', 'T3']] = merged_df[['Fund_Freq','T1', 'T2','T3']].round(3)
    # print(merged_df.head())
    
    # select the first n-1 columns and assign to X dataframe
    X = merged_df.iloc[:, :-1]
    # print(X_uS.head())
    # select the last column and assign to y dataframe
    y = merged_df.iloc[:, -1:]
    
    ########## Scaling section ###################
    # X=StandardScaling(X) # apply to data set a Standard Scaling procedure
    # X=MinMaxScaling(X) # apply to data set a MinMax Scaling procedure
    # X=RobustScaling(X) # apply to data set a Robust Scaling procedure
    # X=UnitVectorScaling(X) # apply to data set a Unit Vector Scaling Scaling procedure
    
    if sel==1:
        ######### Features selection proces ##################
        k=5
        dataset=SelectKBest_cS(X,y,k) # Select the best k features by using chi-Square test
        # dataset=SelectKBest_Anova(X,y,k)  # Select the k best features using the ANOVA F-value test
        # dataset=PCA_fs(X,y,k)   # Reduce the dimensionality of the dataset to k principal components using the PCA algorithm.
    elif sel==0:
        # # data set without features selection
        dataset = pd.concat([X, y], axis=1)
    
    return dataset

# def Data_Set_All(pathBuildings,pathEarthquakes):
def Data_Set_All(pathBuildings):
    allSimRez=Simulation_Data_Cleaning(pathBuildings)
    # allEarthqParam=Data_Set_Earthquakes(pathEarthquakes)

    # Save the DataFrame to a CSV file
    allSimRez.to_csv('DataSet_P3.csv', index=False)  # 'output.csv' is the desired CSV file name
    # Save the DataFrame to a CSV file
    # allEarthqParam.to_csv('AccSet_P3.csv', index=False)  # 'output.csv' is the desired CSV file name

def Find_fft(signal):

        ######## FIND Fundamental frequency for an accelerogram ###############
        # calculate the FFT of the earthquake data
        fft = np.fft.fft(signal)

        # calculate the power spectral density (PSD)
        psd = np.abs(fft) ** 2

        # calculate the frequencies associated with the PSD
        freqs = np.fft.fftfreq(signal.size, 1 / 100)

        # find the index of the maximum amplitude in the PSD
        max_idx = np.argmax(psd)

        # convert the index to a frequency value
        fund_freq = freqs[max_idx]

        # print the fundamental frequency
        # print('Fundamental frequency:', np.abs(fund_freq))
        return(fund_freq)

def scaler_fit(df,scaler):
    # fit and transform the data
    scaled_data = scaler.fit_transform(df)
    # create a new data frame with the scaled data
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df

def fs_fit(X,y,fs_object):
    X_new = fs_object.fit_transform(X, y)
    y=y.values
    return np.concatenate((X_new, y), axis=1)
     
def StandardScaling(df):
    # Standard scaling
    scaler = StandardScaler()
    scaled_df=scaler_fit(df,scaler)
    return scaled_df

def MinMaxScaling(df): 
    # Min-Max Scaling
    scaler = MinMaxScaler()
    scaled_df=scaler_fit(df,scaler)
    return scaled_df

def RobustScaling(df):
    # Robust Scaling
    scaler = RobustScaler()
    scaled_df=scaler_fit(df,scaler)
    return scaled_df

def UnitVectorScaling(df): # create a Normalizer object
    scaler = Normalizer()
    scaled_df=scaler_fit(df,scaler)
    return scaled_df

def SelectKBest_cS(X,y,kp):
    # Select the best k features by using chi-Square test
    selector = SelectKBest(score_func=chi2, k=kp)
    return fs_fit(X,y,selector)
    
def SelectKBest_Anova(X,y,kp):
    # Select the k best features using the ANOVA F-value test
    selector = SelectKBest(score_func=f_classif, k=kp)
    return fs_fit(X,y,selector)
    
def PCA_fs(X,y,k):
    # Reduce the dimensionality of the dataset to k principal components using the PCA algorithm.
    pca = PCA(n_components=k)
    return fs_fit(X,y,pca)

def create_manual_histogram(DataSet, column_intervals, output_file):
    # Încărcați datele din fișierul CSV într-un DataFrame
    df = pd.read_csv(DataSet)

    
    
    # Calculați noile coloane
    df['PGA_of_the_recording_scale1'] = df['PGA_of_the_recording1'] * df['Scale_factor']
    df['PGA_of_the_recording_scale2'] = df['PGA_of_the_recording2'] * df['Scale_factor']
    df_LShape = df[(df['Lshape'] == 1)]
    # df_LShape = df

    results = []
    for column, intervals in column_intervals:
        # Extrageți valorile din coloană
        values = df_LShape[column]

        if isinstance(intervals, int):
            # Creați histograma folosind numărul specificat de intervale
            hist, bin_edges = np.histogram(values, bins=intervals)
        else:
            # Rotunjiți intervalele la o zecimală
            intervals_rounded = [round(val, 1) for val in intervals]
            # Creați histograma folosind intervalele rotunjite
            hist, bin_edges = np.histogram(values, bins=intervals_rounded)

        # Construiți șirul de caractere cu intervalele de bins separate de "&"
        intervale_str = "&".join([f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)])

        # Construiți șirul de caractere cu valorile din fiecare interval separate de "&"
        valori_str = "&".join(map(str, hist))

        # Adăugați datele pentru fiecare interval la rezultate
        results.append([column, intervale_str, valori_str])

        # Creați histograma folosind matplotlib
        plt.hist(values, bins=bin_edges, edgecolor='black')

        # Adăugați etichete cu frecvența deasupra fiecărei bare a histogramului
        for i in range(len(bin_edges) - 1):
            plt.text(
                (bin_edges[i] + bin_edges[i + 1]) / 2,
                hist[i],
                str(hist[i]),
                ha='center',
                va='bottom'
            )

    # Salvare într-un fișier CSV fără antet
    df_to_save = pd.DataFrame(results, columns=['Column', 'Intervale', 'Valori'])
    df_to_save.to_csv(output_file, header=False, index=False)

def extract_data_for_hist(DataSet,columns_to_extract,output_file):
        # Încărcați datele din fișierul CSV într-un DataFrame
    df = pd.read_csv(DataSet)

    # Delete rows where 'DI_cladire' is less than 0 or greater than 2.5
    df_LShape = df[(df['Lshape'] == 1)]
    # df_LShape = df

        # Create a new DataFrame with the extracted columns
    extracted_columns = df_LShape[columns_to_extract]

    extracted_columns.to_csv(output_file, index=False)


