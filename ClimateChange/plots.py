import matplotlib.pyplot as plt
import correlazione
import detrend
import read
import numpy as np

gf = read.read_estati()
detrended_gf, _ = detrend.detrend_signal(gf['Anomaly'])
detrended_gf2 = detrend.block_detrend(gf['Anomaly'], 30)
diff_gf = correlazione.derivata_dx(detrended_gf)
diff_gf2 = correlazione.derivata_dx(detrended_gf2)

fig, axs = plt.subplots(2, 1, figsize=(12, 10))  # 2 plot uno sopra l'altro

# # Primo scatter plot
axs[0].scatter(detrended_gf, diff_gf, s=13, color='blue', label='Dataset B')
axs[0].set_xlabel('Anomalie di Temperatura (°C)', fontsize=20)
axs[0].set_ylabel('derivata destra (°C/s)', fontsize=22)
axs[0].legend(loc='best', fontsize=15)
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
axs[0].tick_params(axis='both', which='major', labelsize=22)
axs[0].set_xticks(np.arange(-1, 1.6, 0.5))
# Secondo scatter plot
axs[1].scatter(detrended_gf2, diff_gf2, s=13, color='red', label='Dataset B')
axs[1].set_xlabel('Anomalie di Temperatura (°C)', fontsize=20)
axs[1].set_ylabel('derivata destra (°C/s)', fontsize=22)
axs[1].legend(loc='best', fontsize=15)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
axs[1].tick_params(axis='both', labelsize=22)
axs[1].set_xticks(np.arange(-1, 1.6, 0.5))
plt.savefig('scplot.pdf')
plt.show()
plt.tight_layout()


# ef = read.read_estati()
# detrended_ef = detrend.detrend_signal(ef['Anomaly'])
# diff_ef = correlazione.derivata_dx(detrended_ef)


#r_s, indici = correlazione.clean_pearson(detrended_gf)
r_s =[-0.6598142488682142, -0.6372596430490364, -0.7199753661467414, -0.6465985284636131, -0.7227258415940981, -0.6658297271718642, -0.6442343040242089, -0.7093650556767934, -0.7317504943126762, -0.7770388794287426, -0.6733786681895452, -0.6652828154447499, -0.6555049816720425, -0.7336837086198182, -0.5453691223742076]
r_s2 = [-0.5253227653953364, -0.6052403558388053, -0.6381547167656602, -0.6462218143687402, -0.7013730665511475, -0.6551262563451699, -0.5896587620059586, -0.5477390296884804, -0.631955303407869, -0.7807826058257593, -0.6553891766898373, -0.6709457023518819, -0.6230036134040516, -0.5857446809773689, -0.5435756411000606]
colors = ['lightyellow', 'lemonchiffon', 'khaki', 'gold', 'yellow']
#-0.6253691223742076
#-0.6235756411000606
plt.figure(figsize=(10, 6))  # Crea una nuova figura per il box plot
# Primo box plot per r_s
box1 = plt.boxplot(r_s, positions=[1], widths=0.6, patch_artist=True, boxprops={'linewidth':1.5}, whiskerprops={'linewidth':1.5}, capprops={'linewidth':1.5})
for patch, color in zip(box1['boxes'], colors):
    patch.set_facecolor(color)
plt.plot(1, r_s[14], 'bo', label='Coeff. XX secolo')  # Evidenzia il valore specifico in r_s
plt.axhline(y=r_s[14], color='b', linestyle='-.', alpha=0.5)

# Secondo box plot per r_s2
box2 = plt.boxplot(r_s2, positions=[2], widths=0.6, patch_artist=True, boxprops={'linewidth':1.5}, whiskerprops={'linewidth':1.5}, capprops={'linewidth':1.5})
for patch, color in zip(box2['boxes'], colors):
    patch.set_facecolor(color)
plt.plot(2, r_s2[14], 'ro', label='Coeff. XX secolo')  # Evidenzia il valore specifico in r_s2
plt.axhline(y=r_s2[14], color='r', linestyle='-.', alpha=0.4)

plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.9)
plt.xlim(0.3, 2.7)  # Aggiornato per includere entrambi i boxplot
plt.ylim(-1, -0.4)
plt.ylabel("Coefficiente di Pearson", fontsize=20)
plt.xticks([1, 2], ['Metodo 1', 'Metodo 2'], fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(loc='best', fontsize=15)
plt.savefig('boxplot.pdf')
plt.show()
