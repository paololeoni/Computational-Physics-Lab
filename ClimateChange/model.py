import read
import correlazione
import detrend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def calcola_derivata_destra(signal):
    if isinstance(signal, pd.Series):
        signal_values = signal.to_numpy()  # Converte la Series in un array NumPy
    else:
        signal_values = signal
    return np.diff(signal_values, append=signal_values[-1])

def leggi_simulazioni(file_path):
    with open(file_path, 'r') as file:
        simulazioni = [np.array(list(map(float, line.split()))) for line in file]
    return simulazioni

def calcola_distanza(segno_reale, derivate_reali, simulazione, derivate_simulazione):
    distanze = np.sqrt((segno_reale - simulazione)**2 + (derivate_reali - derivate_simulazione)**2)
    #return np.sum(distanze)
    return np.mean(distanze)
    #return np.median(distanze)
    #return np.std(distanze)

def calcola_distanza_corr(segno_reale, derivate_reali, simulazione, derivate_simulazione):
    r1, _= correlazione.calcola_correlazione(segno_reale, derivate_reali)
    r2, _  = correlazione.calcola_correlazione(simulazione, derivate_simulazione)

    return np.abs(r2-r1)

def trova_migliore_simulazione(segno_reale, derivate_reali, simulazioni):
    distanze_totali = []
    distanze_totali2 = []
    mediana_maggiore = -np.inf  # Inizializza a un valore molto basso
    indice_normale = -1

    for i, simulazione in enumerate(simulazioni):
        derivate_simulazione = calcola_derivata_destra(simulazione)
        #coefficiente = calcola_distanza_corr(segno_reale, derivate_reali, simulazione, derivate_simulazione) #pearson
        distanza = calcola_distanza(segno_reale, derivate_reali, simulazione, derivate_simulazione)
        distanze_totali.append(distanza)
        #distanze_totali2.append(coefficiente) #pearson

        if distanza > mediana_maggiore:
            mediana_maggiore = distanza
            indice_normale = i   


    #indice_migliore = np.argmin(distanze_totali) + np.argmin(distanze_totali2)
    indice_migliore = np.argmin(distanze_totali)
    #return indice_migliore, (distanze_totali[indice_migliore]/distanze_totali[indice_normale]) + distanze_totali2[indice_migliore]
    return indice_migliore, distanze_totali[indice_migliore]
                                             
def test_ks(segnales_reale, migliore_simulazione):

    # Esegui il test KS
    statistic, p_value = ks_2samp(segnales_reale, migliore_simulazione)
    
    # Restituisci solo il p-value
    return p_value, statistic

# Lettura

gf = read.read_HeadCRUT5()
ef = read.read_estati()
df = read.read_BP()
df['Year'] = (df['Year'] * 1000).apply(lambda x: (1950 - x))
df_reversed = df.iloc[::-1].reset_index(drop=True)
filtered_df = df_reversed[df_reversed['Anomaly'] != -999]

# Detrend

detrended_ef, _ = detrend.detrend_signal(ef['Anomaly']) #30
detrended_df, _ = detrend.detrend_signal(filtered_df['Anomaly']) #50
detrended_gf, _ = detrend.detrend_signal(gf['Anomaly']) #20

# Simulazioni Modello

segno_reale = detrended_gf  # Qui dovresti inserire il tuo segnale reale 
derivate_reali = calcola_derivata_destra(segno_reale)

file_path = 'simulations.txt'
simulazioni = leggi_simulazioni(file_path)

indice_migliore, distanza_minima = trova_migliore_simulazione(segno_reale, derivate_reali, simulazioni)

migliore_simulazione = simulazioni[indice_migliore]

print(test_ks(segno_reale, migliore_simulazione))

plt.figure(figsize=(10, 6))

# Plot del segnale reale
plt.plot(segno_reale, label='Segnale Reale', alpha = 0.5, color='blue', linewidth=2)

# Plot della migliore simulazione
plt.plot(migliore_simulazione, label='Migliore Simulazione', color='red', linewidth=2)

plt.title('Confronto tra Segnale Reale e Migliore Simulazione')
plt.xlabel('Indice')
plt.ylabel('Valore')
plt.legend()
plt.show()