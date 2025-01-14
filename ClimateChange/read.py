import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
import random
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

#=========================================
#   FUNZIONI 
#=========================================

def read_BP():
    column_names = ['Year', 'column2', 'Anomaly']
    nf = pd.read_csv('datdata.csv', header=None, names=column_names)
    nf = nf.drop(nf.columns[1], axis=1)
    return nf

def read_estati():
    #column_names = ['Year', 'Anomaly', 'Liminf', 'Limsup']
    nf = pd.read_csv('dat2data.csv')
    #print(nf.columns)
    nf = nf.drop(nf.columns[4:], axis=1)
    nf.columns = ['Year', 'Anomaly', 'Liminf', 'Limsup']
    return nf

def plot_estati():

    ef['Year'] = ef['Year'].astype(int)
    subset_ef = ef[(ef['Year'] >= 500) & (ef['Year'] <= 2000)]
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Year AD", fontsize=10)
    ax2.set_ylabel("Anomalies", fontsize=10)
    ax2.plot((subset_ef['Year']), subset_ef['Anomaly'], "-", color="r", markersize=3)
    plt.tight_layout()
    plt.show()

def read_gmst_volc():
    df = pd.read_csv("gmst_changes_paleo_volcanic_forcings.csv", header=137)
    
    # Elimina le colonne dall'indice 1 al 15
    df.drop(df.columns[1:16], axis=1, inplace=True)
    
    mean_values = []  # Lista per le medie
    rows_to_remove = []  # Lista per le righe da rimuovere
    
    for index, row in df.iterrows():
        # Seleziona valori da colonne 18 a 40 (17:40 in Python)
        values = row.values[16:40]  # Aggiusta l'indice qui
        # Filtra valori <= 1000, ignora NaN o valori infiniti con np.isfinite()
        filtered_values = [v for v in values if v <= 1000 and np.isfinite(v)]
        
        if len(filtered_values) == 0:  # Se non ci sono valori validi
            rows_to_remove.append(index)
        else:
            mean_values.append(np.mean(filtered_values))
    
    df.drop(index=rows_to_remove, inplace=True)  # Rimuove le righe senza valori validi
    df.drop(df.columns[2:], axis=1, inplace=True)  # Tieni solo le prime due colonne e rimuovi le altre
    df['Mean'] = mean_values  # Aggiungi colonna 'Mean'
    df.columns = ['Year', 'Anomaly_nan', 'Anomaly']
    
    return df

def read_HeadCRUT5():
    cf = pd.read_csv('HadCRUT5.csv')
    idx = cf.columns.get_loc("Lower confidence limit (2.5%)")
    cf = cf.drop(cf.columns[idx:], axis=1)
    cf.columns = ['Year', 'Anomaly']
    return cf


def errorbar(ef):
    ef['ErrorUp'] = (ef['Limsup'] - ef['Anomaly']).abs()
    ef['ErrorDown'] = (ef['Anomaly'] - ef['Liminf']).abs()

    # Calcola la media delle lunghezze delle barre di errore
    ef['Error'] = (ef['ErrorUp'] + ef['ErrorDown']) / 2

    # Elimina le colonne non più necessarie
    ef.drop(['Liminf', 'Limsup', 'ErrorUp', 'ErrorDown'], axis=1, inplace=True)  
    return ef

def discrepanza_max(df, liminf, limsup, step):

    df = df.copy()
    df['Year'] = df['Year'].astype(int)
    medie = []
    df2 = df[(df['Year'] >= liminf) & (df['Year'] <= limsup)]
    for start in range(0, len(df2), step):
        end = start + step
        media_blocco = df2['Anomaly'].iloc[start:end].mean()
        medie.append(media_blocco)
    distanza = np.abs(max(medie) - min(medie))
    return distanza

def distanza_max(df, liminf, limsup):

    df = df.copy()
    df['Year'] = df['Year'].astype(int)
    df2 = df[(df['Year'] >= liminf) & (df['Year'] <= limsup)]
    d = abs(df2['Anomaly'].max() - df2['Anomaly'].min())
    return d

def simulazione_discrepanza(df, width, N_simulazioni, liminf, limsup):
    discrepanza_originale = discrepanza_max(df, liminf, limsup, width)
    conteggio_maggiore = 0
    indici_estratti = set()  # Inizializza un set vuoto per tenere traccia degli indici già estratti

    while len(indici_estratti) < N_simulazioni and len(indici_estratti) < len(df) - width:
        # Scegli un indice a caso tra i primi N-width elementi
        indice_casuale = np.random.randint(0, len(df) - width)
        # Controlla se l'indice è già stato estratto; se sì, continua senza fare nulla
        if indice_casuale in indici_estratti:
            continue
        # Aggiungi l'indice al set degli indici estratti
        indici_estratti.add(indice_casuale)

        # Crea un sub-df a partire dall'indice casuale con lunghezza width
        sub_df = df.iloc[indice_casuale:indice_casuale + width]
        # Calcola la distanza massima nel sub-df
        distanza_massima = distanza_max(sub_df, liminf, limsup)

        # Incrementa il conteggio se la distanza massima è maggiore della discrepanza originale
        if distanza_massima > discrepanza_originale:
            conteggio_maggiore += 1

    # Calcola il p-value come la frazione di volte in cui la distanza massima supera la discrepanza originale
    p_value = conteggio_maggiore / len(indici_estratti)

    # Istogramma dei risultati
    plt.hist([1] * conteggio_maggiore + [0] * (len(indici_estratti) - conteggio_maggiore), bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1], ['Distanza ≤ Discrepanza', 'Distanza > Discrepanza'])
    plt.xlabel('Risultato Simulazione')
    plt.ylabel('Conteggio')
    plt.title('Risultati delle Simulazioni')
    plt.show()

    return p_value


def block_mean(arr, years, block_size):

    trim_size = len(arr) - (len(arr) % block_size)
    trimmed_arr = arr[:trim_size]
    trimmed_years = years[:trim_size]
    
    mean_arr = np.mean(trimmed_arr.reshape(-1, block_size), axis=1)
    variance_arr = np.var(trimmed_arr.reshape(-1, block_size), axis=1)

    # Calcola l'anno iniziale per ogni blocco
    block_start_years = trimmed_years[::block_size]
    
    return mean_arr, variance_arr, block_start_years

def plot_differences(series1, series2):
    # Calcola le differenze punto-punto
    differences = series1 - series2
    
    # Crea un DataFrame per facilitare il plotting
    diff_df = pd.DataFrame({
        'Time': series1.index,  # Assumendo che series1 e series2 abbiano gli stessi indici temporali
        'Difference': differences
    })
    
    # Plot delle differenze
    plt.figure(figsize=(10, 6))
    plt.plot(diff_df['Time']+500, diff_df['Difference'], label='Difference between Series1 and Series2')
    plt.xlabel('Time')
    plt.ylabel('Difference')
    plt.title('Point-by-Point Differences between Two Series')
    plt.legend()
    plt.grid(True)
    plt.show()

def cumulazioni(df, nome_colonna_dati, block_size):
    # Utilizza la funzione modificata per ottenere medie e varianze dei blocchi
    mean_arr, variance_arr, block_start_years = block_mean(df[nome_colonna_dati].values, df['Year'].values, block_size)
    
    new_df = pd.DataFrame({
        'block_index': block_start_years,
        'block_mean_temperature_anomaly': mean_arr,
        'block_variance_temperature_anomaly': variance_arr  # Aggiunge la varianza di ciascun blocco al DataFrame
    })
    
    somma_cumulativa = 0
    somma_quadri_cumulativa = 0
    medie_cumulative = []
    varianze_cumulative = []

    for i, (media, varianza) in enumerate(zip(new_df['block_mean_temperature_anomaly'], new_df['block_variance_temperature_anomaly']), 1):
        somma_cumulativa += media
        somma_quadri_cumulativa += media**2
        media_cumulativa = somma_cumulativa / i
        medie_cumulative.append(media_cumulativa)
        
        media_quadri_cumulativa = somma_quadri_cumulativa / i
        varianza_cumulativa = media_quadri_cumulativa - media_cumulativa**2
        varianze_cumulative.append(varianza_cumulativa)
        
        print(f"Media cumulativa dopo {i} numeri: {media_cumulativa}")
        print(f"Varianza cumulativa dopo {i} numeri: {varianza_cumulativa}")

    # Plot della media e varianza cumulativa
    plt.figure(figsize=(12, 8))
    plt.plot(medie_cumulative[1:], label='Media Cumulativa', color='blue')
    plt.plot(varianze_cumulative[1:], label='Varianza Cumulativa', color='red')
    plt.xlabel('Numero di Passi')
    plt.ylabel('Valore Cumulativo')
    plt.title('Media e Varianza Cumulativa in Funzione dei Passi')
    plt.legend()
    plt.show()

    return new_df



#=========================================
#   LETTURA FILE
#=========================================

gf = read_HeadCRUT5()
ef = read_estati()
#print(len(ef['Anomaly']))
df = read_BP()
df['Year'] = (df['Year'] * 1000).apply(lambda x: (1950 - x))
df_reversed = df.iloc[::-1].reset_index(drop=True)
filtered_df = df_reversed[df_reversed['Anomaly'] != -999]
#=========================================
#   STAMPA MEDIA GLOBALE
#=========================================

mf = [gf, ef, filtered_df]
with open('results.txt', 'w') as file:
    file.write("Media dati vulcanici (dal 850 ad oggi) :" + str(gf['Anomaly'].mean()) + "\n" +
               "Media Estati (dal 500 ad oggi) :" + str(ef['Anomaly'].mean()) + "\n" +
               "Media Paleo (da 783k anni fa ad oggi) :" + str(filtered_df['Anomaly'].mean()) + "\n")

#=========================================
#   STAMPA DISCREPANZE
#=========================================
    
lista_dataframe = [
    (gf, 1000, 2000),
    (ef, 1000, 2000),
    (filtered_df, -781050, -7050)
]

steps = [10, 30, 50, 100]
titoli = ['Vulcani', 'estivo', 'paleo']
#print(discrepanza_max(filtered_df, -781050, -7050, 100))
for i, (df, liminf, limsup) in enumerate(lista_dataframe):
    with open('results.txt', 'a') as file:
        file.write(f"\nDataset " + titoli[i] + ':\n')
    for step in steps:
        discrepanza = discrepanza_max(df, liminf, limsup, step)
        with open('results.txt', 'a') as file:
            file.write(f"Step {step}: Discrepanza massima = {discrepanza}\n")
            #file.write("-" * 30)

#with open('results.txt', 'a') as file:
    #file.write('\np value = ' + str(simulazione_discrepanza(ef, 50, 100000, 1000, 2000)))
#print(simulazione_discrepanza(ef,20, 1449, 500, 2000))


#==================================
#   SMOOTHING CON MEDIE A BLOCCHI
#==================================
#ef, block_size = 30 -> parametri simulazia: numpoints=49, mean=-0.1, sigma=0.2
# block_size = 30
# subset_ef = ef[(ef['Year'] >= 500) & (ef['Year'] <= 2000)]
# new_ef = cumulazioni(ef, 'Anomaly', block_size)
"""""
#==================================
#   PLOT del DS RIDOTTO
#==================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot((new_ef['block_index']), new_ef['block_mean_temperature_anomaly'], "-", color="r", markersize=2)
ax.plot((ef['Year']), ef['Anomaly'], "-", color="g", markersize=2)
ax.set_xlabel("Year BC", fontsize=10)
ax.set_ylabel("Anomalies", fontsize=10)
plt.tight_layout()
plt.show()
"""
"""
#==================================
#   PLOTTING DATI PALEO

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot((filtered_df['Year']), filtered_df['Anomaly'], "-", color="r", markersize=2)
ax.set_xlabel("Year BC", fontsize=10)
ax.set_ylabel("Anomalies", fontsize=10)
plt.tight_layout()
plt.show()
"""

#===================================
#   PLOTTING DATI ESTIVI
# ef['Year'] = ef['Year'].astype(int)
# subset_ef = ef[(ef['Year'] >= 500) & (ef['Year'] <= 2000)]
# fig2 = plt.figure(figsize=(10, 6))
# ax2 = fig2.add_subplot(111)
# ax2.set_xlabel("Year AD", fontsize=10)
# ax2.set_ylabel("Anomalies", fontsize=10)
# ax2.plot((subset_ef['Year']), subset_ef['Anomaly'], "-", color="r", markersize=3)
# plt.tight_layout()
# plt.show()

#===================================
#   PLOTTING DATI VULCANICI
#gf['Year'] = gf['Year'].astype(int)
#subset_gf = gf[(gf['Year'] >= 850) & (gf['Year'] <= 1000)]  
#fig3 = plt.figure(figsize=(10, 6))
#ax3 = fig3.add_subplot(111)
#ax3.set_xlabel("Year AD", fontsize=10)
#ax3.set_ylabel("Anomalies", fontsize=10)
#ax3.plot((subset_gf['Year']), subset_gf['Anomaly'], "-", color="g", markersize=2)
#plt.tight_layout()
#plt.show()

"""""
#===================================
#   OVERALL PLOTTING
#===================================
"""   
"""""     
mf = [filtered_df, ef, gf]
titoli = ['Anomalie Temperatura: 800.000 anni fa', 'Anomalie Temperatura: Estati 2000 anni fa', 'Anomalie Temperatura + vulcani: 2000 anni fa']
colori = ['b','g','r']
fig = plt.figure(figsize=(10,6)) #figure identifica la finestra in cui c'è il grafico, figsize cambia la dimensione del canvas
n_plots = 3

for i, mf in enumerate(mf, start=1):
    ax = fig.add_subplot(2, 2, i)  # Crea un subplot in una griglia 2x2
    ax.plot(mf['Year'], mf['Anomaly'], "-", color=colori[i-1], markersize=3)
    ax.set_xlabel("Years", fontsize=10)
    ax.set_ylabel("Anomalies", fontsize=10)
    ax.set_title(titoli[i-1])

plt.tight_layout()
plt.show()


"""


#======================================
#   SIMU RW GAUSS
#======================================


# print(len(ef['Anomaly']))
# print(ef['Anomaly'].mean())
# print(ef['Anomaly'].std())
# print(ef['Anomaly'].head(5))
# new_ef['block_mean_temperature_anomaly'].to_csv('means.txt', index=False, header=False)



"""""
# Leggere il file generato dal programma C++, rw con feedback
simulations_data = []
with open("simulations.txt", "r") as file:
    for line in file:
        points = np.array([float(x) for x in line.split()])
        simulations_data.append(points)

ks_results = []
for simulated_anomalies in simulations_data:
    # Calcola il test di Kolmogorov-Smirnov tra i dati empirici e quelli simulati
    ks_statistic, p_value = stats.ks_2samp(ef['Anomaly'], simulated_anomalies)
    ks_results.append((ks_statistic, p_value, simulated_anomalies))

# Trovare la simulazione con il miglior p-value
best_simulation_statistic, best_simulation_p_value, best_simulation_data = max(ks_results, key=lambda x: x[1])
print("\n" + "p value = " + str(best_simulation_p_value))
random_element = random.choice(simulations_data)

fig, bx= plt.subplots(figsize=(10, 6))
#fic, cx = plt.subplots(figsize=(10, 6))

#cx.plot(ef['Year'], random_element, color = "y") #questo plot non arriva fino a 2000 anni perchè plotta il numero di punti
#cx.plot(ef['Year'], ef['Anomaly'], "-", color="g", alpha=0.5 , markersize=2)
#cx.set_xlabel("Step", fontsize=10)
#cx.set_ylabel("Anomalies", fontsize=10)
#cx.set_title('Random')
bx.plot(new_ef['block_index'], best_simulation_data, color = "b") #questo plot non arriva fino a 2000 anni perchè plotta il numero di punti
bx.plot(new_ef['block_index'], new_ef['block_mean_temperature_anomaly'], "-", color="g", markersize=2)
#bx.plot(ef['Year'], ef['Anomaly'], "-", alpha = 0.5, color="g", markersize=2)
bx.set_xlabel("Step", fontsize=10)
bx.set_ylabel("Anomalies", fontsize=10)
#cx.set_title('Best')

plt.tight_layout()
plt.show()
plt.show()
#plot_differences(ef['Anomaly'], best_simulation_data)

"""""
# Leggere il file generato dal programma C++, generate  walks
# simulations_data = []
# with open("simu-blocks.txt", "r") as file:
#     for line in file:
#         points = np.array([float(x) for x in line.split()])
#         simulations_data.append(points)

# ks_results = []
# for simulated_anomalies in simulations_data:
#     # Calcola il test di Kolmogorov-Smirnov tra i dati empirici e quelli simulati
#     ks_statistic, p_value = stats.ks_2samp(ef['Anomaly'], simulated_anomalies)
#     ks_results.append((ks_statistic, p_value, simulated_anomalies))

# # Trovare la simulazione con il miglior p-value
# best_simulation_statistic, best_simulation_p_value, best_simulation_data = max(ks_results, key=lambda x: x[1])
# random_element = simulations_data[np.random.randint(len(simulations_data))]
# random_p_value = stats.ks_2samp(ef['Anomaly'], random_element)[1]

# # Best p-value
# print("\n" + "best p value = " + str(best_simulation_p_value))

# #Random p-value
# print("\n" + "random p value = " + str(random_p_value))

# fig = plt.figure(figsize=(10,6)) 
# bx = fig.add_subplot(111)
# fig, bx= plt.subplots(figsize=(10, 6))
# fic, cx = plt.subplots(figsize=(10, 6))

# # Random plot
# cx.plot(ef['Year'], random_element, color = "b") #questo plot non arriva fino a 2000 anni perchè plotta il numero di punti
# cx.plot(ef['Year'], ef['Anomaly'], "-", color="g", alpha=0.5 , markersize=2)
# cx.set_xlabel("Step", fontsize=10)
# cx.set_ylabel("Anomalies", fontsize=10)
# cx.set_title('Random')

# # Best plot
# bx.plot(ef['Year'], best_simulation_data, color = "b") #questo plot non arriva fino a 2000 anni perchè plotta il numero di punti
# bx.plot(ef['Year'], ef['Anomaly'], alpha = 0.5, color = "g") #questo plot non arriva fino a 2000 anni perchè plotta il numero di punti
# bx.set_xlabel("Step", fontsize=10)
# bx.set_ylabel("Anomalies", fontsize=10)
# bx.set_title('Best')

    # plt.tight_layout()
# plt.show()
