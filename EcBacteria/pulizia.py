# import pandas as pd
# import numpy as np

# def lista_famiglie(df):

#     #   divide per famiglie come prima cosa (gruppi)
#     #   ritorna una array di famiglie

#     gruppi = df.groupby(['Pos', 'Channel'])
#     lista_gruppi = [gruppo for _, gruppo in gruppi]
#     return lista_gruppi, gruppi

# def seleziona_famiglia(df, pos, ch):
#     lista_gruppi, gruppi = lista_famiglie(df)
#     indici = {chiave: indice for indice, (chiave, _) in enumerate(gruppi)}
#     indice_famiglia = indici[(pos, ch)]
#     famiglia_specifica = lista_gruppi[indice_famiglia]
#     return famiglia_specifica

# def find_mother_son_pairs2(df):

#     mini_dfs = []
#     for index1, row1 in df.iterrows():
#         for index2, row2 in df.iterrows():
#             if index1 != index2 and row1['Cell'] == row2['MothCell'] :    
#                 if abs(row2['Divfr'] - row1['Birthfr']) < 2:
#                     # row2 is the mother and row1 is the son
#                     mini_df = pd.DataFrame([row2, row1], columns=df.columns)
#                     mini_dfs.append(mini_df)

#     return mini_dfs

# # divisione x famiglie
# df = pd.read_csv("Datasetgrande_Batteri.csv",names = ['Pos','Channel','Cell','MothCell','Birthfr','Divfr','Td','Tri1','Tri2','Tri3','Trt','Tn','Tc','Lb','Lri1','Lri2','Lri3','Lrt','Ln','Lc','Ld','Time'],header=0,delimiter=r'\s+')
# lista_gruppi, gruppi = lista_famiglie(df)

# # selezione della famiglia da analizzare 
# that_familia = seleziona_famiglia(df,9,1)
# #print(that_familia)

# # mini_dataframes = find_mother_son_pairs2(that_familia)
# # for i, mini_df in enumerate(mini_dataframes):
# #     print(f"Mini-DF {i+1}:")
# #     print(mini_df)
# #     print()


# all_mini_dfs = []
# # # Apply the find_mother_son_pairs function to each group
# # for index, gruppo in enumerate(lista_gruppi):  
# #     mini_dataframes = find_mother_son_pairs2(gruppo)
# #     all_mini_dfs.extend(mini_dataframes)

# #print(len(all_mini_dfs))

# # Printing all the mini DataFrames
# # for i, mini_df in enumerate(all_mini_dfs):
# #     print(f"Mini-DF {i+1}:")
# #     print(mini_df)
# #     print()

# data_for_new_df = []

# def comma_to_float(value):
#     if isinstance(value, str):
#         return float(value.replace(',', '.'))
#     return float(value)  # If it's already a float or int, just convert to float directly


# # Extract specific values from each mini DataFrame
# for mini_df in all_mini_dfs:
#     if len(mini_df) == 2:  # Ensure there are exactly two rows (mother and son)
#         lip = comma_to_float(mini_df.iloc[0]['Lri1'])  # 'Lri1' from first row (mother)
#         li = comma_to_float(mini_df.iloc[1]['Lri1'])   # 'Lri1' from second row (son)
#         lb = comma_to_float(mini_df.iloc[1]['Lb'])     # 'Lb' from second row (son)
#         ld = comma_to_float(mini_df.iloc[1]['Ld'])     # 'Ld' from second row (son)
#         data_for_new_df.append([lip, li, lb, ld])

# # Create the new DataFrame      
# new_df = pd.DataFrame(data_for_new_df, columns=['Lip', 'Li', 'Lb', 'Ld'])
# #print(new_df['Li'])
# corr_var = new_df.corr()
# #print("Correlation matrix of the variables:\n", corr_var)

# std_lb = new_df['Lb'].std()
# std_ld = new_df['Ld'].std()
# std_lp = new_df['Lip'].std()
# std_li = new_df['Li'].std()

# # Print the standard deviation
# # print(f"The standard deviation of 'Lb' is: {std_lb}")
# # print(f"The standard deviation of 'Lb' is: {std_ld}")
# # print(f"The standard deviation of 'Lb' is: {std_lb}")
# # print(f"The standard deviation of 'Lb' is: {std_lb}")


# def mother_son_pairs(df):
#     pairs = []
#     for indice, row in df.iterrows():
#         mother = row['MothCell']
#         pos = row['Pos']
#         channel = row['Channel']
#         birthFr = row['BirthFr']
#         mothers = df[(df['Pos'] == pos) & (df['Channel'] == channel) & (df['Cell'] == mother) & (abs(df['DivFr'] - birthFr) <= 5)]
#         if (not mothers.empty) & (len(mothers) == 1):
#             pairs.append((mothers.iloc[0], row))
#         elif (len(mothers) > 1):
#             print("piu' madri trovate")
#     return pairs            

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

df = pd.read_csv("Datasetgrande_Batteri.csv", names=['Pos', 'Channel', 'Cell', 'MothCell', 'Birthfr', 'Divfr', 'Td', 'Tri1', 'Tri2', 'Tri3', 'Trt', 'Tn', 'Tc', 'Lb', 'Lri1', 'Lri2', 'Lri3', 'Lrt', 'Ln', 'Lc', 'Ld', 'Time'], header=0, delimiter=r'\s+')

def mother_son_pairs(df):
    pairs = []
    count = 0
    for indice, row in df.iterrows():
        mother = row['MothCell']
        pos = row['Pos']
        channel = row['Channel']
        birthFr = row['Birthfr']
        mothers = df[(df['Pos'] == pos) & (df['Channel'] == channel) & (df['Cell'] == mother) & (abs(df['Divfr'] - birthFr) <= 2)]
        if (not mothers.empty) & (len(mothers) == 1):
            pairs.append([mothers.iloc[0], row])
        elif (len(mothers) > 1):
            count += 1
    print('{} doppie possibili madri trovate'.format(count))
    return pairs

pairs = mother_son_pairs(df)
print(len(pairs))

Lip_array = []
Li_array = []
Lb_array = []
Ld_array = []
Td_array = []
Ti_array = []

for (mother, son) in pairs:
    Lip_array.append(mother['Lri1'])
    Li_array.append(son['Lri1'])
    Lb_array.append(son['Lb'])
    Ld_array.append(son['Ld'])
    Td_array.append(son['Td'])
    Ti_array.append(son['Tri1'])

# Crea un DataFrame con le quattro liste come colonne
df = pd.DataFrame({'Lip': Lip_array, 'Li': Li_array, 'Lb': Lb_array, 'Ld': Ld_array, 'Td': Td_array, 'Tri1': Ti_array})

# Funzione per rimuovere le virgolette e sostituire le virgole con i punti nei decimali
def clean_value(value):
    if isinstance(value, str):
        return value.replace('"', '').replace(',', '.')
    else:
        return value

# Applica la funzione a tutto il DataFrame
df = df.applymap(clean_value)
# Salva il DataFrame in un file CSV
df.to_csv('Datipulitifig6.csv', index=False)

Lip_float = [float(value.replace(',', '.')) for value in Lip_array]

#Lip_float = list(map(float, Lip_array))

