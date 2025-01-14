import pandas as pd
import numpy as np

N = 100  # Numero di iterazioni
#CONDIZIONARE SU LI NELLE SIMULAZIONI
def condition_on(dataframe, variable, value, range_width):
    return dataframe[(dataframe[variable] > value - range_width) & (dataframe[variable] < value + range_width)]
for _ in range(N):
    n = 1000000
    #DEFINIZIONE FILE
    df = pd.read_excel("DatipulitiAcetato.xlsx",names = ['Lip','Li','Lb','Ld','C','D'],header = 1)

    df = df.astype(float)
    print("Reale = ", df['Lip'].corr(df['Ld']))
    df_cond = condition_on(df,'Li', np.median(df['Li']), np.std(df['Li'])/np.sqrt(len(df['Li'])))
    print("RC = ", df_cond['Lip'].corr(df['Ld']))


    '''DEFINIZIONE PARAMETRI per SIMULAZIONI'''

    #Lip
    Lipmedia = df['Lip'].mean()
    Lipdev_std = df['Lip'].std()
    etaip_gaussiani = np.random.normal(Lipmedia, Lipdev_std, n)

    #G RATE
    g_rate = (np.log(df['Ld']/df['Lb']))/df['D']

    g_ratemedia = g_rate.mean()
    g_ratedev_std = g_rate.std()
    g_rate_gaussiani = np.random.normal(g_ratemedia, g_ratedev_std, n)

    #LB
    c_d = df['D']-df['C']
    cdmedia = c_d.mean()
    cddev_std = c_d.std()
    c_d_gaussiani = np.random.normal(abs(cdmedia), abs(cddev_std), n)

    delta_ii1 =  df['Li'] - df['Lip']
    deltaiimedia = delta_ii1.mean()
    deltaiistd = delta_ii1.std()
    delta_ii = np.random.normal(abs(deltaiimedia), abs(deltaiistd), n)

    etabmedia = df['Lb'].std()
    etabdev_std = ((df['Lb']-df['Lb'].mean())/len(df['Lb'])).mean()
    etab_gaussiani = np.random.normal(abs(etabmedia), abs(etabdev_std), n)

    etaimedia = df['Li'].std()
    etaidev_std = ((df['Li']-df['Li'].mean())/len(df['Li'])).mean()
    etai_gaussiani = np.random.normal(abs(etaimedia), abs(etaidev_std), n)

    etadmedia = df['Ld'].std()
    etaddev_std = ((df['Ld']-df['Ld'].mean())/len(df['Ld'])).mean()
    etad_gaussiani = np.random.normal(abs(etadmedia), abs(etaddev_std), n)

    etabdmedia  = np.sqrt(etabmedia*2+etadmedia*2)
    etabddev_std = np.sqrt(etabdev_std*2+etaddev_std*2)
    etabd_gaussiani = np.random.normal(abs(etabdmedia), abs(etabddev_std), n)

    delta_bd = (df['Ld'].mean()-df['Lb'].mean())

    #EXP
    etac_dmedia = (c_d-c_d.mean()).mean()
    etac_ddevstd = ((c_d-c_d.mean())/len(c_d)).std()
    etacd_gaussiani = np.random.normal(abs(etac_dmedia), abs(etac_ddevstd), n)

    exp_alpha = np.exp(g_rate_gaussiani*c_d_gaussiani)
    exp_beta = np.exp(g_rate_gaussiani*c_d_gaussiani+etacd_gaussiani)
    #BETA
    Lip_beta = etaip_gaussiani
    Lb_beta = Lip_beta*exp_alpha+etab_gaussiani
    Li_beta = (Lip_beta+delta_ii)/2+etai_gaussiani
    Ld_beta = pd.Series([max(x, y) for x, y in zip(Lb_beta+delta_bd+etabd_gaussiani, Li_beta*exp_beta)])

    #ALPHA
    Lip_alpha = etaip_gaussiani
    Lb_alpha = Lip_alpha*exp_alpha+etab_gaussiani
    Li_alpha = (Lip_alpha+delta_ii)/2+etai_gaussiani
    Ld_alpha = 2*Li_alpha*exp_alpha+etai_gaussiani

    dfalpha = pd.DataFrame({'Lip': Lip_alpha, 'Li': Li_alpha, 'Lb': Lb_alpha, 'Ld': Ld_alpha})

    correlationalpha = dfalpha['Lip'].corr(dfalpha['Ld'])
    print("Simulata alpha = ", correlationalpha)
   # print("Correlazione Lb Ld alpha: ",correlationalpha)
    # Calcola la matrice di correlazione
    #correlation_matrixalpha = dfalpha.corr()

    #print("Matrice di correlazione alpha:")
    #print(correlation_matrixalpha)

    dfbeta = pd.DataFrame({'Lip': Lip_beta, 'Li': Li_beta, 'Lb': Lb_beta, 'Ld': Ld_beta})

    correlationbeta = dfbeta['Lip'].corr(dfbeta['Ld'])
    print("Simulata beta = ", correlationbeta)


    alpha_cond = condition_on(dfalpha,'Li', np.median(dfalpha['Li']), np.std(dfalpha['Li'])/np.sqrt(len(dfalpha['Li'])))

    correlationalphacond = alpha_cond['Lip'].corr(alpha_cond['Ld'])
    print("SC alpha = ", correlationalphacond)


    beta_cond = condition_on(dfbeta,'Li', np.median(dfbeta['Li']), np.std(dfbeta['Li'])/np.sqrt(len(dfbeta['Li'])))
    correlationalbetacond = beta_cond['Lip'].corr(beta_cond['Ld'])
    print("SC beta = ", correlationalbetacond)

    
 






