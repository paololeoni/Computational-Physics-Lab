import read
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import kstest


#=======================================
#   DETREND
#=======================================
ef = read.read_estati()
gf = read.read_BP()
gf['Year'] = (gf['Year'] * 1000).apply(lambda x: (1950 - x))
gf_reversed = gf.iloc[::-1].reset_index(drop=True)
filtered_gf = gf_reversed[gf_reversed['Anomaly'] != -999]

# num_blocks_list = [5, 10, 20, 30, 50]

# fig, axs = plt.subplots(len(num_blocks_list), 1, figsize=(10, 6 * len(num_blocks_list)))

# if not isinstance(axs, np.ndarray):
#     axs = [axs]

# for i, num_blocks in enumerate(num_blocks_list, start=0):
    
#     print(f"\nAnalizzando {num_blocks} blocchi:")
#     blocks = np.array_split(gf['Anomaly'].values, num_blocks)

#     detrended_signal = np.array([])
#     r_squared_values = []
#     trends = np.empty_like(gf['Anomaly'])


#     start = 0
#     # Per ogni blocco
#     for block in blocks:

#         # Calcola il trend lineare
#         x = np.arange(len(block))
#         slope, intercept = np.polyfit(x, block, 1)
#         trend = slope * (x) + intercept
#         trends[start:start + len(block)] = trend
#         start += len(block)     
#         # Sottrai il trend
#         detrended_block = block - trend
        
#         # Aggiungi il blocco detrended alla lista
#         detrended_signal = np.concatenate([detrended_signal, detrended_block])


#     # Crea un subplot per ciascuna misura di blocco
#     ax = plt.subplot(len(num_blocks_list), 1, i+1)
#     ax.plot(gf['Anomaly'].values, alpha=0.5, label="Original Data")
#     ax.plot(trends, label=f"Fit for {num_blocks} blocks", color='r')
#     ax.set_title(f"{num_blocks} Blocks")
#     ax.legend()
#     axs[i].plot(gf['Anomaly'].values, label='Original Signal', alpha=0.5, color="g")
#     axs[i].plot(detrended_signal, label='Detrended Signal', color="r")
#     axs[i].set_title(f'Original vs Detrended Signal for {num_blocks} Blocks')
#     axs[i].legend()



# plt.show()
# plt.tight_layout()

# ===========================================
#   PARTE 2
# ===========================================

def detrend_signal(signal):
    n = len(signal)
    trend = np.zeros(n)
    start_index = 0

    while start_index < n:
        for end_index in range(start_index + 2, n + 1):
            # Prendi il segmento corrente del segnale
            segment = signal[start_index:end_index]
            X = sm.add_constant(np.arange(len(segment)))  # Aggiungi una colonna di 1 per l'intercetta
            model = sm.OLS(segment, X).fit()
            p_value = model.f_pvalue
            
            # Controlla il p-value per determinare se continuare con l'attuale segmento o iniziarne uno nuovo
            if p_value < 0.05 and end_index - start_index > 2:  # Inizia un nuovo segmento se p < 0.05, ma ignora per i primi 2 punti
                break
        else:
            # Se non si entra nel 'if', significa che siamo riusciti a includere tutti i punti rimanenti nel segmento
            end_index = n + 1
        
        # Fit finale per il segmento corrente
        segment = signal[start_index:end_index - 1]
        X = sm.add_constant(np.arange(len(segment)))
        model = sm.OLS(segment, X).fit()
        trend_line = model.predict(X)
        
        # Salva il trend calcolato
        trend[start_index:end_index - 1] = trend_line
        
        start_index = end_index - 1  # Aggiorna l'indice di partenza per il nuovo segmento

    detrended_signal = signal - trend
    return detrended_signal, trend

uf, trend = detrend_signal(ef['Anomaly'])
# plt.figure(figsize=(10, 6))
# plt.plot(ef['Anomaly'], label='Original Signal', marker='o')
# plt.plot(uf, label='Detrended Signal')
# plt.plot(trend, label='Trend', linestyle=':')
# plt.legend()
# plt.title('Signal Detrending')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.show()


# PARTE 3
def block_detrend(signal, block_length):  
    
    num_full_blocks = len(signal) // block_length

    # Tronca il segnale per eliminare l'ultimo blocco incompleto se necessario
    trimmed_signal_length = num_full_blocks * block_length
    trimmed_signal = signal[:trimmed_signal_length]

    # Dividi il segnale troncato in blocchi
    blocks = np.array_split(trimmed_signal, num_full_blocks)

    detrended_signal = np.array([])
    trends = np.empty_like(trimmed_signal)

    start = 0
    for block in blocks:
        # Calcola il trend lineare
        x = np.arange(len(block))
        slope, intercept = np.polyfit(x, block, 1)
        trend = slope * x + intercept
        trends[start:start + len(block)] = trend
        start += len(block)
        # Sottrai il trend
        detrended_block = block - trend
        # Aggiungi il blocco detrended alla lista
        detrended_signal = np.concatenate([detrended_signal, detrended_block])

    # Plot dei risultati
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Segnale Originale', alpha=0.5, color="g")
    plt.plot(np.arange(trimmed_signal_length), detrended_signal, label='Segnale Detrended', color="r")
    plt.plot(np.arange(trimmed_signal_length), trends, label='Trend', color='b', linestyle='--')
    plt.title(f'Detrending con Blocchi di Lunghezza {block_length}')
    plt.legend()
    plt.show()
    return detrended_signal


def test_blocks_normality(signal, block_length):
    # Calcola il numero di blocchi completi e tronca il segnale se necessario
    num_full_blocks = len(signal) // block_length
    trimmed_signal_length = num_full_blocks * block_length
    trimmed_signal = signal[:trimmed_signal_length]

    # Dividi il segnale troncato in blocchi
    blocks = np.array_split(trimmed_signal, num_full_blocks)
    
    # Inizializza un array vuoto per i p-values
    p_values = []

    # Esegue il test KS su ciascun blocco
    for block in blocks:
        # Calcola media e deviazione standard del blocco
        mean = np.mean(block)
        std = np.std(block)
        
        # Esegue il test KS confrontando il blocco con una distribuzione normale parametrizzata
        _, p_value = kstest(block, 'norm', args=(mean, std))
        p_values.append(p_value)
    
    return np.array(p_values)

def test_blocks_cov(signal1, signal2, block_length):
    # Calcola il numero di blocchi completi e tronca il segnale se necessario
    num_full_blocks = len(signal1) // block_length
    trimmed_signal_length = num_full_blocks * block_length
    trimmed_signal1 = signal1[:trimmed_signal_length]
    trimmed_signal2 = signal2[:trimmed_signal_length]

    # Dividi il segnale troncato in blocchi
    blocks1 = np.array_split(trimmed_signal1, num_full_blocks)
    blocks2 = np.array_split(trimmed_signal2, num_full_blocks)

    # Inizializza un array vuoto per le covarianze
    covarianze = []

    # Calcola la covarianza per ciascuna coppia di blocchi
    for block1, block2 in zip(blocks1, blocks2):
        # Calcola la matrice di covarianza tra i due blocchi
        cov_matrix = np.cov(block1, block2)
        
        # Seleziona la covarianza tra i due blocchi dalla matrice di covarianza
        # La covarianza tra block1 e block2 è in posizione [0, 1] nella matrice
        cov = cov_matrix[0, 1]
        
        # Aggiunge il valore di covarianza alla lista
        covarianze.append(cov)
    
    return np.array(covarianze)

def test_blocks_std(signal1, signal2, block_length):
    # Calcola il numero di blocchi completi e tronca il segnale se necessario
    num_full_blocks = len(signal1) // block_length
    trimmed_signal_length = num_full_blocks * block_length
    trimmed_signal1 = signal1[:trimmed_signal_length]
    trimmed_signal2 = signal2[:trimmed_signal_length]

    # Dividi il segnale troncato in blocchi
    blocks1 = np.array_split(trimmed_signal1, num_full_blocks)
    blocks2 = np.array_split(trimmed_signal2, num_full_blocks)

    # Inizializza un array vuoto per i prodotti delle deviazioni standard
    std_products = []

    # Calcola il prodotto delle deviazioni standard per ciascuna coppia di blocchi
    for block1, block2 in zip(blocks1, blocks2):
        # Calcola la deviazione standard per ogni blocco
        std1 = np.std(block1, ddof=1)  # ddof=1 per la deviazione standard campionaria
        std2 = np.std(block2, ddof=1)  # ddof=1 per la deviazione standard campionaria
        
        # Calcola il prodotto delle due deviazioni standard
        std_product = std1 * std2
        
        # Aggiunge il valore del prodotto alla lista
        std_products.append(std_product)
    
    return np.array(std_products)
# signal = np.random.random(1000)  # Esempio di segnale
# block_length = 100
# p_values = test_blocks_normality(signal, block_length)
# print(p_values)
