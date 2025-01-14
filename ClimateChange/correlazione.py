# SCATTER PLOT CORRELAZIONE, DISTRIBUZIONI ANOMALIE E DERIVATE
# SEGNALE DETRENDED
import read
import detrend
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.stats import chi2_contingency, norm, chi2

# def detrend_signal(gv, num_intervals):

#     detrended_signal_list = []  # Usa una lista per raccogliere i blocchi detrended
#     r_squared_values = []
#     trends = np.empty_like(gv)
#     start = 0

#     blocks = np.array_split(gv.values, num_intervals)

#     for block in blocks:
#         # Calcola il trend lineare
#         x = np.arange(len(block))
#         slope, intercept = np.polyfit(x, block, 1)
#         trend = slope * x + intercept  # Nota: l'uso di (x) o x è equivalente
#         trends[start:start + len(block)] = trend
#         start += len(block)  
        
#         # Sottrai il trend
#         detrended_block = block - trend
#         detrended_signal_list.append(detrended_block)  # Aggiungi il blocco detrended alla lista

#         # Calcolo del coefficiente R^2
#         ss_res = np.sum((block - trend) ** 2)  # Somma dei quadrati dei residui
#         ss_tot = np.sum((block - np.mean(block)) ** 2)  # Somma dei quadrati totali
#         r_squared = 1 - (ss_res / ss_tot)  # Calcolo di R^2
#         r_squared_values.append(r_squared)

#     # Concatena i blocchi detrended per ottenere il segnale detrended completo
#     detrended_signal = np.concatenate(detrended_signal_list)
#     return detrended_signal

def derivata_dx(signal):

    derivata = np.diff(signal) / 1
    
    derivata = np.append(derivata, derivata[-1])  # Estende l'array della derivata

    return derivata

def correlazione_spearman(signal1, signal2):

    # Assicurati che entrambi i segnali abbiano la stessa lunghezza
    if len(signal1) != len(signal2):
        raise ValueError("I segnali devono avere la stessa lunghezza.")

    # Calcola il coefficiente di correlazione di Pearson
    coefficiente_r, p_value = spearmanr(signal1, signal2)

    return coefficiente_r, p_value

def correlazione_pearson(signal1, signal2):

    # Assicurati che entrambi i segnali abbiano la stessa lunghezza
    if len(signal1) != len(signal2):
        raise ValueError("I segnali devono avere la stessa lunghezza.")

    # Calcola il coefficiente di correlazione di Pearson
    coefficiente_r, p_value = pearsonr(signal1, signal2)

    return coefficiente_r, p_value

def plot_4(signal1, signal2):
    # Istogramma bidimensionale
    plt.figure(figsize=(10, 7))

    # Primo subplot per lo scatter plot
    plt.subplot(2, 2, 1)  # 2 rows, 2 columns, first subplot
    lunghezza = len(signal1)  # Calcola la lunghezza delle serie

    # Definisci i punti da plottare in blu (i primi fino a lunghezza - 50)
    x_blu = signal1[:lunghezza-30]
    y_blu = signal2[:lunghezza-30]

    # Definisci i punti da plottare in arancione (gli ultimi 30)
    x_arancione = signal1[lunghezza-30:]
    y_arancione = signal2[lunghezza-30:]

    # Plotta i punti in blu
    plt.scatter(x_blu, y_blu, alpha=0.7, s=10, color='blue', label='Punti precedenti')

    # Plotta gli ultimi 50 punti in arancione
    plt.scatter(x_arancione, y_arancione, alpha=0.7, s=10, color='orange', label='Ultimi 50 punti')
    plt.title('Scatter Plot')
    plt.xlabel('Anomalie')
    plt.ylabel('Derivata')

    # Secondo subplot per l'istogramma bidimensionale
    plt.subplot(2, 2, 2)  # 2 rows, 2 columns, second subplot
    plt.hist2d(signal1, signal2, bins=30, cmap='Blues')
    plt.colorbar().set_label('Densità')
    plt.title('Istogramma Bidimensionale')
    plt.xlabel('Anomalie')
    plt.ylabel('Derivata')

    # Istogramma Signal 1
    plt.subplot(2, 2, 3)  # 2 rows, 2 columns, third subplot
    plt.hist(signal1, bins='sturges', alpha=0.7, color='blue')
    plt.title('Istogramma Anomalie')
    plt.xlabel('Valore')
    plt.ylabel('Frequenza')

    # Istogramma Signal 2
    plt.subplot(2, 2, 4)  # 2 rows, 2 columns, fourth subplot
    plt.hist(signal2, bins='sturges', alpha=0.7, color='green')
    plt.title('Istogramma Derivata')
    plt.xlabel('Valore')
    plt.ylabel('Frequenza')

    plt.tight_layout()
    plt.show()

def clean_pearson(signal, segment_length=100):
    # Calcolo la derivata dx per tutto il segnale.
    dx = derivata_dx(signal)

    # Calcola il numero di elementi da rimuovere per avere un array che sia un multiplo di segment_length.
    to_remove = len(signal) % segment_length
    signal_adjusted = signal[to_remove:]   
    derivata_dx_adjusted = dx[to_remove:]
    
    # Calcola il numero di segmenti (escluso l'ultimo).
    num_segments =  int(len(signal_adjusted) / segment_length)
    pearson_coefficients = []
    ps =[]
    indici=[]
    # Calcola il coefficiente di Pearson per ciascun segmento, escluso l'ultimo.
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        indici.append(start_idx)
        segment = signal_adjusted[start_idx:end_idx]
        derivata_segment = derivata_dx_adjusted[start_idx:end_idx]
        
        # Calcolo del coefficiente di Pearson per il segmento.
        coeff, p_value = pearsonr(segment, derivata_segment)
        pearson_coefficients.append(coeff)
        ps.append(p_value)
    #pearson_coefficients.pop()
    return (pearson_coefficients), indici, ps

def t_test(signal, r):  

    t_value = (r * np.sqrt(len(signal)- 3)) / np.sqrt(1 - r**2)

    # Calcola i gradi di libertà
    df = len(signal) - 3

    # Calcola il valore critico della distribuzione t di Student per un livello di significatività del 5%
    # Si può modificare il parametro `alpha` per un diverso livello di significatività
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, df)

    print(f"Valore t calcolato: {t_value}")
    print(f"Valore critico t per un livello di significatività del {alpha*100}%: {t_critical}")

    # Confronta il valore t calcolato con il valore critico
    if abs(t_value) > t_critical:
        print("esiste una correlazione significativa a questo livello di significatività. L'ipotesi nulla viene respinta.")
    else:
        print("Non c'è una differenza significativa. L'ipotesi nulla non viene respinta.")


# ef = read.read_estati()
# nef = read.errorbar(ef)
# detrended_ef, _ = detrend.detrend_signal(ef['Anomaly'])
# detrended_ef2 = detrend.block_detrend(ef['Anomaly'], 30)

# diff_ef = derivata_dx(detrended_ef)
# diff_ef2 = derivata_dx(detrended_ef2)
# cov = detrend.test_blocks_cov(detrended_ef, diff_ef, 100)
# std = detrend.test_blocks_std(detrended_ef, diff_ef, 100)
# print(cov)
# print(std)
# ks_statistic, p_value = stats.kstest(diff_ef2, 'norm', args=(diff_ef2.mean(), diff_ef2.std()))
# print(p_value)

# r, p = correlazione_spearman(detrended_ef, diff_ef)
# r2, p2 = correlazione_pearson(detrended_ef2, diff_ef2)
# #print(r2)
# #print(r2, p2, r, p)

# m = detrend.test_blocks_normality(detrended_ef, 100)
# n = detrend.test_blocks_normality(detrended_ef2, 100)
# o = detrend.test_blocks_normality(diff_ef, 100)
# q = detrend.test_blocks_normality(diff_ef2, 100)
# print(m)
# print(n)
# print(o)
# print(q)

#print(r,p)

#t_test(detrended_ef, r)
# print("p-value e correlazione dati ESTIVI:\n")
# print(p,r)
# plot_4(detrended_ef, diff_ef)
# test_chi2(detrended_ef, 12)



# df = read.read_BP()
# df['Year'] = (df['Year'] * 1000).apply(lambda x: (1950 - x))
# df_reversed = df.iloc[::-1].reset_index(drop=True)
# filtered_df = df_reversed[df_reversed['Anomaly'] != -999]
# detrended_df, _ = detrend.detrend_signal(filtered_df['Anomaly'])
# detrended_df2 = detrend.block_detrend(ef['Anomaly'], 45)
# # ks_statistic, p_value = stats.kstest(detrended_df, 'norm', args=(detrended_df.mean(), detrended_df.std()))
# # print(p_value)
# diff_df = derivata_dx(detrended_df)
# diff_df2 = derivata_dx(detrended_df2)
# # ks_statistic, p_value = stats.kstest(diff_df, 'norm', args=(diff_df.mean(), diff_df.std()))
# # print(p_value)
# #print(detrended_df.mean(), detrended_df.std(), detrended_df[(len(detrended_df)-5):])
# r, p = correlazione_spearman(detrended_df, diff_df)
# r2, p2 = correlazione_pearson(detrended_df2, diff_df2)
# #print(r2, p2, r, p)

#print(r,p)
#t_test(detrended_df, r)
# print("\np-value e correlazione dati PALEO:\n")
# print(p,r)
# plot_4(detrended_df, diff_df)
# test_chi2(detrended_df,11)



# gf = read.read_HeadCRUT5()
# detrended_gf, _ = detrend.detrend_signal(gf['Anomaly'])
# detrended_gf2 = detrend.block_detrend(ef['Anomaly'], 20)
# # ks_statistic, p_value = stats.kstest(detrended_gf, 'norm', args=(detrended_gf.mean(), detrended_gf.std()))
# # print(p_value)
# diff_gf = derivata_dx(detrended_gf)
# diff_gf2 = derivata_dx(detrended_gf2)
# # ks_statistic, p_value = stats.kstest(diff_gf, 'norm', args=(diff_gf.mean(), diff_gf.std()))
# # print(p_value)
# #print(detrended_gf.mean(), detrended_gf.std(), detrended_gf[:5])
# r, p = correlazione_spearman(detrended_gf, diff_gf)
# r2, p2 = correlazione_pearson(detrended_gf2, diff_gf2)
# #print(r2, p2, r, p)
# #print(r,p)
# #t_test(detrended_gf, r)
# # print("\np-value e correlazione dati RECENTI:\n")
# # print(p,r)
# # plot_4(detrended_gf, diff_gf)
# # #test_chi2(detrended_gf,9 )



# # ANALISI 2


# diff_ef_new = derivata_dx(detrended_ef[:(len(detrended_ef)-100)])
# #r, _ = calcola_correlazione(detrended_ef[:(len(detrended_ef)-100)],diff_ef_new)
# r_s, indici, p = clean_pearson(detrended_ef)
# #print(r_s, p)


#assert len(detrended_ef) >= 100 * len(r_s), "Il segnale non è sufficientemente lungo."

# Ciclo che itera sui coefficienti

# for i, coefficient in enumerate(r_s):
#     # Calcoliamo l'inizio e la fine del segmento del segnale da considerare in questa iterazione
#     start = i * 100
#     end = start + 100
    
#     # Estraiamo il segmento del segnale
#     signal_segment = detrended_ef[start:end]
    
#     # Assumiamo che tu abbia una funzione calculate_t_student che fa ciò che serve
#     t_student = t_test(signal_segment, coefficient)
    
#     # Stampa il risultato (o fai quello che ti serve con esso)
#     print(f"Iterazione {i+1}, Coefficiente: {coefficient}, Valore t-Student: {t_student}")

# print(r)
# print(r_s)
# print(indici)
# print(len(r_s), len(indici))


# Prima figura: Box plot
# plt.figure(figsize=(10, 6))  # Crea una nuova figura per il box plot
# plt.boxplot(r_s, positions=[1], widths=0.6)

# # Aggiungi i coefficienti come punti colorati
# plt.plot(1, r, 'ro', label='Coeff. ultimi TOT anni')
# plt.plot(1, r_s[len(r_s)-2], 'bo', label='Coeff. penultimi TOT anni')
# plt.plot(1, r_s[len(r_s)-3], 'mo', label='Coeff. terzultimi TOT anni')
# plt.plot(1, r_s[len(r_s)-4], 'go', label='Coeff. quartultimi TOT anni')

# # Aggiungi linee orizzontali per facilitare il confronto visivo
# plt.axhline(y=r, color='r', linestyle='--')
# plt.axhline(y=r_s[len(r_s)-2], color='b', linestyle='--')
# plt.axhline(y=r_s[len(r_s)-3], color='m', linestyle='--')
# plt.axhline(y=r_s[len(r_s)-4], color='g', linestyle='--')

# plt.xticks([1], ['Coeff. Pearson Random'])
# plt.ylabel('Coefficiente di Pearson')
# plt.title('Confronto Coefficiente di Pearson')
# plt.legend()

# Seconda figura: Scatter plot
# plt.figure(figsize=(10, 6))  # Crea una nuova figura per lo scatter plot
# plt.scatter(np.arange(len(r_s)), r_s, s=13, color='b')
# plt.title('Scatter Plot')
# plt.xlabel('Anomalie')
# plt.ylabel('Derivata')

# plt.show()