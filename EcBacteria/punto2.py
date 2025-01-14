import pandas as pd
import  numpy as np
import pulizia

def remove_columns(df, columns_to_remove):
    """
    Elimina le colonne specificate da un DataFrame.

    Parametri:
        df (pd.DataFrame): Il DataFrame da cui rimuovere le colonne.
        columns_to_remove (list): Una lista di stringhe che rappresenta i nomi delle colonne da rimuovere.

    Ritorna:
        pd.DataFrame: Un nuovo DataFrame senza le colonne specificate.
    """
    return df.drop(columns=columns_to_remove)

def reorder_columns(df, new_order):
    """
    Riordina le colonne di un DataFrame secondo l'ordine specificato.

    Args:
    df (pd.DataFrame): DataFrame da riordinare.
    new_order (list): Lista contenente i nomi delle colonne nell'ordine desiderato.

    Returns:
    pd.DataFrame: Nuovo DataFrame con le colonne riordinate.
    """
    return df[new_order]

def comma_to_float(value):
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)  # If it's already a float or int, just convert to float directly

def mother_son_pairs(df):
    pairs = []
    for indice, row in df.iterrows():
        mother = row['MothCell']
        pos = row['Pos']
        channel = row['Channel']
        birthFr = row['Birthfr']
        mothers = df[(df['Pos'] == pos) & (df['Channel'] == channel) & (df['Cell'] == mother) & (abs(df['Divfr'] - birthFr) <= 5)]
        if (not mothers.empty) & (len(mothers) == 1):
            pairs.append((mothers.iloc[0], row))
        elif (len(mothers) > 1):
            print("piu' madri trovate")
    return pairs

file_names = ["Datasetgrande_Batteri.csv", "Dati_Acetato.csv", "Dati_Alanina.csv", "Dati_Glicerolo.csv", "Dati_Glocerolo1.csv", "Dati_Glucosio.csv", "Dati_Mannosio.csv"]

# Lista per tenere traccia dei DataFrame letti
dataframes = []

# Itera su ogni file
for file_name in file_names:
    # Leggi il DataFrame dal file corrente
    if file_name=="Datasetgrande_Batteri.csv":
        df = pd.read_csv(file_name, names=['Pos','Channel','Cell','MothCell','Birthfr','Divfr','Td','Tri1','Tri2','Tri3','Trt','Tn','Tc','Lb','Lri1','Lri2','Lri3','Lrt','Ln','Lc','Ld','Time'], header=0, delimiter=r'\s+')
    df = pd.read_csv(file_name, names=['Pos','Channel','Cell','MothCell','Birthfr','Divfr','Td','Tri1','Tri2','Tri3','Trt','Tn','Tc','Lb','Lri1','Lri2','Lri3','Lrt','Ln','Lc','Ld'], header=0, delimiter=r'\s+')
    # Aggiungi il DataFrame alla lista

    dataframes.append(df)

#remove_columns(dataframes[0], 'Time')
reorder = dataframes[0].columns.tolist()
modified_dfs = [reorder_columns(df, reorder) for df in dataframes[1:]]

combined_df = pd.concat(modified_dfs, ignore_index=True)
pairs = mother_son_pairs(combined_df)
print(pairs)

lista_gruppi, gruppi = pulizia.lista_famiglie(combined_df)

all_mini_dfs = []
# Apply the find_mother_son_pairs function to each group
for _,gruppo in gruppi:  
    mini_dataframes = pulizia.find_mother_son_pairs2(gruppo)
    all_mini_dfs.extend(mini_dataframes)

#print(all_mini_dfs[len(all_mini_dfs)-3])

data_for_new_df = []
# Extract specific values from each mini DataFrame
for mini_df in all_mini_dfs:
    if len(mini_df) == 2:  # Ensure there are exactly two rows (mother and son)
        lip = comma_to_float(mini_df.iloc[0]['Lri1'])  # 'Lri1' from first row (mother)
        li = comma_to_float(mini_df.iloc[1]['Lri1'])   # 'Lri1' from second row (son)
        lb = comma_to_float(mini_df.iloc[1]['Lb'])     # 'Lb' from second row (son)
        ld = comma_to_float(mini_df.iloc[1]['Ld'])     # 'Ld' from second row (son)
        data_for_new_df.append([lip, li, lb, ld])

# Create the new DataFrame      
new_df = pd.DataFrame(data_for_new_df, columns=['Lip', 'Li', 'Lb', 'Ld'])
#print(len(new_df['Lip']))
corr_var = new_df.corr()
#print("Correlation matrix of the variables:\n", corr_var)
