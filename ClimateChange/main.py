import numpy as np
import matplotlib.pyplot as plt
import correlazione

# Leggi il file
with open('data.txt', 'r') as file:
    data = file.readlines()

# Converti ogni riga in un array numpy e poi in una lista di array
data = [np.fromstring(line, dtype=float, sep=' ') for line in data]

# Converti la lista di array in un unico array numpy
data_array = np.array(data)
derivata_array = np.array([correlazione.derivata_dx(simulation) for simulation in data_array])

# Seleziona una simulazione casuale
random_simulation_index = np.random.randint(0, len(data))
random_simulation = data_array[random_simulation_index]

coefficients_array = []
indici_array = []
p_values_array = []

for simulation in data_array:
    coeffs, indici, ps = correlazione.clean_pearson(simulation)
    coefficients_array.append(coeffs)
    indici_array.append(indici)
    p_values_array.append(ps)

coefficients_array = np.array(coefficients_array)
last_coeffs_simulated = coefficients_array[:, -1]

# Calcola il p-value
more_extreme = np.sum(np.abs(last_coeffs_simulated) >= np.abs(-0.5453691223742076))
p_value = more_extreme / 10000

print(f"p-value: {p_value}")

# Interpretazione del p-value
if p_value < 0.05:  # Assumi una soglia di significatività di 0.05
    print("Il risultato empirico è statisticamente significativo; è improbabile che sia dovuto al caso.")
else:
    print("Non ci sono prove sufficienti per considerare il risultato empirico significativamente diverso da quanto ci si potrebbe aspettare sotto l'ipotesi nulla.")

# Crea il plot
plt.figure(figsize=(10, 6))
plt.plot(random_simulation)
plt.title(f'Simulazione Casuale #{random_simulation_index + 1}')
plt.xlabel('Numero di Estrazione')
plt.ylabel('Valore')
plt.grid(True)
plt.show()

