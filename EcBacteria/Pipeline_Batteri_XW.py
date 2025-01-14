import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Datasetgrande_Batteri.csv",names = ['Pos','Channel','Cell','MothCell','Birthfr','Divfr','Td','Tri1','Tri2','Tri3','Trt','Tn','Tc','Lb','Lri1','Lri2','Lri3','Lrt','Ln','Lc','Ld','Time'],header=0,delimiter=r'\s+')

df_filtrato = df[['Pos','Channel','Cell','MothCell','Lb', 'Lri1', 'Lri2', 'Lri3', 'Ld']]

# Utilizza il metodo .loc[] per assegnare i valori
df_filtrato.loc[:, 'Lb'] = df_filtrato['Lb'].str.replace(',', '.').astype(float)
df_filtrato.loc[:, 'Lri1'] = df_filtrato['Lri1'].str.replace(',', '.').astype(float)
df_filtrato.loc[:, 'Lri2'] = df_filtrato['Lri2'].str.replace(',', '.').astype(float)
df_filtrato.loc[:, 'Lri3'] = df_filtrato['Lri3'].str.replace(',', '.').astype(float)
df_filtrato.loc[:, 'Ld'] = df_filtrato['Ld'].str.replace(',', '.').astype(float)

colonna_riordinata = df['Cell'].sort_values()

if(df_filtrato['Pos']==5, df_filtrato['Channel']==1, df_filtrato['MothCell']==1):
    print(colonna_riordinata.tail(20))






#print(df)

#corrispondenze diagramma dataset:
    #Lb -> Lb
    #Li(lunghezza all'inizio della replicazione del DNA) 
    #Lip(lunghezza all'inizio della replicazione del DNA nel ciclo precedente)
    #Li e Lip di riferiscono a Lri1,Lri2 e Lri3 che corrispondono a:
    #Lri1(lunghezza all'inizio della replicazione del DNA)
    #Lri2(lunghezza della prima cellula figlia all'inizio della replicazione del DNA)
    #Lri3(lunghezza della seconda cellula figlia all'inizio della replicazione del DNA)
    #Ld -> Ld

# Compute correlations
corr_df = df_filtrato.corr()


#print("Correlation Matrix :\n", corr_df)


# Function to filter data around a given value within a specified range
def condition_on(dataframe, variable, value, range_width=0.4):
    return dataframe[(dataframe[variable] > value - range_width) & (dataframe[variable] < value + range_width)]

#condiziona su z nel modello alpha
df_cond = condition_on(df_filtrato, 'Lri2', 2.5)
#matrice di correlazione di alpha condizionando su x
corr_condx = df_cond.corr()
'''
#condiziona su w conseguentemente a x su alpha
df_ALPHA_cond = condition_on(df_ALPHA_cond, 'W', 0)
#matrice di correlazione di alpha dopo aver condizionato anche su w
corr_ALPHA_condxw = df_ALPHA_cond.corr()
'''

#print("Matrice di correlazione di alpha dopo aver condizionato su Lri2 \n",corr_condx)


'''
#condiziona su x nel modello alpha
df_BETA_cond = condition_on(df_BETA, 'X', 0)
#matrice di correlazione di alpha condizionando su x
corr_BETA_condx = df_BETA_cond.corr()
#condiziona su w conseguentemente a x su alpha
df_BETA_cond = condition_on(df_BETA_cond, 'W', 0)
#matrice di correlazione di alpha dopo aver condizionato anche su w
corr_BETA_condxw = df_BETA_cond.corr()

print("Matrice di correlazione di beta dopo aver condizionato su x \n",corr_BETA_condx)
print("Matrice di correlazione di beta dopo aver condizionato su x e poi su w \n",corr_BETA_condxw)
'''

    