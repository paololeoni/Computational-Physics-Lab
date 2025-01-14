import pandas as pd
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


def condition_on(dataframe, variable, value, range_width=0.5):
    return dataframe[(dataframe[variable] > value - range_width) & (dataframe[variable] < value + range_width)]
    

df = pd.read_excel("DatipulitiGlicerolo.xlsx",header = 0)
df.dropna(subset=['Lip', 'Lb'], inplace=True)  # Rimuove righe con valori NaN
df = df[(np.abs(df['Lip']) < 1e10) & (np.abs(df['Lb']) < 1e10)] 


correlation_matrix = df.corr()
# print(correlation_matrix)
# print(correlation_matrix2)

# MEDIA E STD DEL DF
df = df.astype(float)
medie = df.mean()
deviazioni_std = df.std()
# print("Medie:")
# print(medie)
# print("\nDeviazioni standard:")
# print(deviazioni_std)

'''DEFINIZIONE COSTANTI'''
eta_ip = (df['Lip']-medie['Lip'])/len(df['Lip']) + 1  
eta_b = (df['Lb']-medie['Lb'])/len(df['Lb'])
eta_i = (df['Li']-medie['Li'])/len(df['Li'])
eta_d = (df['Ld']-medie['Ld'])/len(df['Ld'])

# C+D, ETA_{C+D}
cd = abs(df['Td'] - df['Tri1']) 
cd_mean = cd.mean()
eta_cd = (cd -  cd_mean)/len(cd)

etac_dmedia = (cd-cd.mean()).mean()
etac_ddevstd = ((cd-cd.mean())/len(cd)).std()
etacd_gaussiani = np.random.normal(abs(etac_dmedia), abs(etac_ddevstd), len(df))


# LAMBDA
g_rate = (np.log(df['Ld']/df['Lb']))/df['Td']
# EXP(C+D)
# Lb_modified = df['Lb']
# A = np.reshape(df['Lip'], (len(df['Lip']), 1))
# m, residuals, rank, s = lstsq(A, Lb_modified)
# exp1 = m[0] #0.8454637339842321 0.7957173792709318
# exp2 = medie['Lb'] / medie['Lip']
EXP_BETA = np.exp(g_rate*cd + etacd_gaussiani)
EXP_ALPHA = np.exp(g_rate*cd)
#print(exp1, exp2)

# DELTA_ii
Li_mod = 2*(df['Li']) - df['Lip']
A = np.ones((len(df['Lip']), 1))
q, residuals, rank, s = np.linalg.lstsq(A, Li_mod, rcond=None)
delta_ii1 = q[0] # 1.4574054366685938 1.4574054366685942
delta_ii2 = 2 * medie['Li'] - medie['Lip']
#print(delta_ii1,delta_ii2)

# DELTA_bd
d = df['Ld'] -df['Lb']
delta_bd = d.mean(skipna=True)

# ETA bd
etabmedia = df['Lb'].std()
etabdev_std = ((df['Lb']-df['Lb'].mean())/len(df['Lb'])).mean()
etab_gaussiani = np.random.normal(abs(etabmedia), abs(etabdev_std), len(df))
etadmedia = df['Ld'].std()
etaddev_std = ((df['Ld']-df['Ld'].mean())/len(df['Ld'])).mean()
etad_gaussiani = np.random.normal(abs(etadmedia), abs(etaddev_std), len(df))
etabdmedia  = np.sqrt(etabmedia*2+etadmedia*2)
etabddev_std = np.sqrt(etabdev_std*2+etaddev_std*2)
etabd_gaussiani = np.random.normal(abs(etabdmedia), abs(etabddev_std), len(df))

condition = (df['Li'] * np.exp(EXP_BETA) >= df['Lb'] + delta_bd + etabd_gaussiani)
succ = condition.sum()  # Conta i True
ins = (~condition).sum()  # Conta i False

print(succ/len(df['Li']), ins/len(df['Li']))

# SIMULAZIONI
# sim_alpha = pd.DataFrame(columns=['Lip', 'Li', 'Lb', 'Ld'])

# # Assegna i valori di 'Lip' per sim_alpha
# sim_alpha['Lip'] = eta_ip

# # Calcola i valori di 'Lb' per sim_alpha
# sim_alpha['Lb'] = df['Lip'] * EXP_ALPHA + eta_b

# # Calcola i valori di 'Li' per sim_alpha
# sim_alpha['Li'] = (df['Lip'] + delta_ii1) / 2 + eta_i

# # Calcola i valori di 'Ld' per sim_alpha
# sim_alpha['Ld'] = 2 * (df['Li'] * EXP_ALPHA ) + eta_d


# # Costruisci sim_beta
# sim_beta = pd.DataFrame(columns=['Lip', 'Li', 'Lb', 'Ld'])

# # Assegna i valori di 'Lip' per sim_beta
# sim_beta['Lip'] = sim_alpha['Lip']

# # Assegna i valori di 'Lb', 'Li', e 'Ld' per sim_beta
# sim_beta['Lb'] = df['Lip'] * EXP_ALPHA + eta_b
# sim_beta['Li'] = sim_alpha['Li']
# delta = medie['Ld'] - medie['Lb']
# sim_beta['Ld'] = np.maximum(df['Lb']+delta+eta_bd,df['Li']*EXP_BETA)

# corr_alpha = sim_alpha.corr()
# corr_beta = sim_beta.corr()
# corr_real = df.corr()


# gf = condition_on(df, 'Li', 0.7, 0.25)
# print("ALPHA\n",corr_alpha)
# print("\nBETA\n",corr_beta)

# # PLOT
# asse_x = 'Lb'
# asse_y = 'Ld'
# # Crea il grafico
# plt.figure(figsize=(8, 6))

# # Scatter plot principale, più marcato
# plt.scatter(df[asse_x], df[asse_y], color='blue', alpha=1.0, label=asse_x)

# # Scatter plot secondario, più affievolito
# plt.scatter(gf[asse_x], gf[asse_y], color='red', alpha=0.5, label=asse_y)

# # Aggiungi dettagli al grafico
# plt.title('Confronto tra Due Scatter Plots')
# plt.xlabel(asse_x)
# plt.ylabel(asse_y)

# # Mostra il grafico
# plt.show()
