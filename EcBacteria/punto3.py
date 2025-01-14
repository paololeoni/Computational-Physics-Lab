import pandas as pd
from itertools import chain
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr, shapiro

def custom_filter(df):
    # Calcola i quantili senza includere potenziali NaN
    percentile95_Lri1 = df['Lri1'].dropna().quantile(0.95)
    condition_Lri1 = df['Lri1'] <= percentile95_Lri1

    log_expr = (np.log(df['Ld']/df['Lb']))/df['Td']
    percentile95_Td = log_expr.dropna().quantile(0.95)

    # Utilizza AND logico e escludi righe con NaN in colonne critiche
    df_filtered = df[condition_Lri1]

    # Calcola i percentili sul DataFrame filtrato
    percentile95_Ld = df_filtered['Ld'].quantile(0.95)
    percentile95_Lb = df_filtered['Lb'].quantile(0.95)

    condition_Ld = df_filtered['Ld'] <= percentile95_Ld
    condition_Lb = df_filtered['Lb'] <= percentile95_Lb

    df_filtered_Ld = df_filtered[condition_Ld]
    df_filtered_Lb = df_filtered[condition_Lb]

    return df_filtered, df_filtered_Ld, df_filtered_Lb

def linreg_res(dfs):
    residuals_Lb_Lri1 = []
    residuals_Ld_Lri1 = []

    # Regressioni e Calcolo residui
    for df in dfs:
        df_filtered, df_Ld_filtered, df_Lb_filtered = custom_filter(df)
        #print(len(df_filtered), len(df_Ld_filtered), len(df_Lb_filtered))
        
        # Calcola i residui per Lb su Lri1 nel DataFrame filtrato per Lb
        if not df_Lb_filtered.empty:
            X_Lri1_Lb = df_Lb_filtered[['Lri1']]
            y_Lb = df_Lb_filtered['Lb']
            # df = df.dropna(subset=['Lri1'])
            # X_Lri1_Lb = df[['Lri1']]
            # y_Lb = df['Lb']
            model_Lb = LinearRegression().fit(X_Lri1_Lb, y_Lb)
            predictions_Lb = model_Lb.predict(X_Lri1_Lb)
            residuals_Lb = y_Lb - predictions_Lb
            residuals_Lb_Lri1.append(list(residuals_Lb))
        
        # Calcola i residui per Ld su Lri1 nel DataFrame filtrato per Ld
        if not df_Ld_filtered.empty:
            X_Lri1_Ld = df_Ld_filtered[['Lri1']]
            y_Ld = df_Ld_filtered['Ld']
            # X_Lri1_Ld = df[['Lri1']].dropna()
            # y_Ld = df['Ld']
            model_Ld = LinearRegression().fit(X_Lri1_Ld, y_Ld)
            predictions_Ld = model_Ld.predict(X_Lri1_Ld)
            residuals_Ld = y_Ld - predictions_Ld
            residuals_Ld_Lri1.append(list(residuals_Ld))

    # Appiattisci le liste di liste
    flat_residuals_Lb = list(chain.from_iterable(residuals_Lb_Lri1))
    flat_residuals_Ld = list(chain.from_iterable(residuals_Ld_Lri1))

    # Verifica la lunghezza e tronca se necessario per assicurarti che siano della stessa lunghezza
    min_length = min(len(flat_residuals_Lb), len(flat_residuals_Ld))
    flat_residuals_Lb = flat_residuals_Lb[:min_length]
    flat_residuals_Ld = flat_residuals_Ld[:min_length]

    # Plot residui
    for index, (residuals_Lb, residuals_Ld) in enumerate(zip(residuals_Lb_Lri1, residuals_Ld_Lri1)):
        # Assicurati che le lunghezze siano uguali per il plotting
        min_length = min(len(residuals_Lb), len(residuals_Ld))
        residuals_Lb = np.array(residuals_Lb[:min_length])
        residuals_Ld = np.array(residuals_Ld[:min_length])

        correlation, p_value = spearmanr(residuals_Lb, residuals_Ld)
        sw_statistic, sw_p_value = shapiro(residuals_Lb)
        sw_statistic2, sw_p_value2 = shapiro(residuals_Ld)
        print("r e p df: " + str (index))
        print(correlation, p_value)
        # print("p shapiro df: ",index)
        # print(sw_p_value, sw_p_value2)
        
        # Crea una figura per ogni DataFrame
        plt.figure(figsize=(10, 6))
        plt.scatter(residuals_Lb, residuals_Ld, color='blue', label='Residui')
        
        # Calcolo della regressione lineare per la retta
        model = LinearRegression().fit(residuals_Lb.reshape(-1, 1), residuals_Ld)
        line_x = np.linspace(residuals_Lb.min(), residuals_Lb.max(), 100)
        line_y = model.predict(line_x.reshape(-1, 1))
        print("Coefficient angolare (slope) della retta di regressione:", model.coef_[0])

        
        # Plot della retta di regressione
        plt.plot(line_x, line_y, color='red', label='Fit lineare')
        plt.xlabel(r'Residui di Lb [$\mu$m]', fontsize=20)
        plt.ylabel(r'Residui di Ld [$\mu$m]', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=15) 
        plt.legend(loc='best', fontsize=18)
        plt.grid(True)
        plt.savefig('mannosio.pdf', format='pdf', bbox_inches='tight')
        plt.show()

def GR(dfs):

    gr=[]
    for df in dfs:
        appo = (np.log(df['Ld']/df['Lb']))/df['Td']
        gr.append(appo.mean())

    return gr

# Lista dei file CSV
file_list = ['acetato.csv', 'alanina.csv', 'glicerolo.csv', 'glicerolo1.csv', 'glucosio.csv', 'mannosio.csv']  # Aggiungi qui i tuoi file

# Colonne che vuoi leggere da ogni file
colonne_da_mantenere = ['Lb', 'Lri1', 'Ld', 'Td'] # Sostituisci con i nomi delle colonne reali

# Lista per raccogliere i DataFrame
dfs = []

# Lettura files
for file in file_list:

    df = pd.read_csv(file, header=0)
    df_dropped = df[colonne_da_mantenere]
    dfs.append(df_dropped)
    #print(df['Td'])

# Regressioni e Residui
linreg_res(dfs)

# Grow Rates
g_rate = GR(dfs)
print(g_rate)


