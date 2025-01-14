import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Settings
n = 833  # Number of data points
mu_X, sigma2_x = 0, 1
coefficients = {'aY': 0, 'bXY': 5, 'aZ': 0, 'bXZ': 5, 'aW': 0, 'bZW': 5, 'bYW': 5}

# Generate X
X = np.random.normal(mu_X, np.sqrt(sigma2_x), n)
#X = np.random.exponential(10, n)

# Generate Y and Z for both models
Y = coefficients['aY'] + coefficients['bXY'] * (X - mu_X) + np.random.normal(0, 1, n)
Z = coefficients['aZ'] + coefficients['bXZ'] * (X - mu_X) + np.random.normal(0, 1, n)

# Generate W for ALPHA and BETA
W_ALPHA = coefficients['aW'] + coefficients['bZW'] * (Z - np.mean(Z)) + np.random.normal(0.5, 0.1, n)
W_BETA = coefficients['aW'] + coefficients['bZW'] * (Z - np.mean(Z)) + coefficients['bYW'] * (Y - np.mean(Y)) + np.random.normal(0.5, 0.1, n)

# Create dataframes
df_ALPHA = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'W': W_ALPHA})
df_BETA = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'W': W_BETA})

# Compute correlations
corr_ALPHA = df_ALPHA.corr()
corr_BETA = df_BETA.corr()

print("Correlation Matrix for ALPHA Model:\n", corr_ALPHA)
print("\nCorrelation Matrix for BETA Model:\n", corr_BETA)

# Function to filter data around a given value within a specified range
def condition_on(dataframe, variable, value, range_width=0.5):
    return dataframe[(dataframe[variable] > value - range_width) & (dataframe[variable] < value + range_width)]
def condition_dataframe_on_ranges(df, condition_vars, operators, condition_values):
    """
    Condiziona il dataframe 'df' sulle variabili specificate in 'condition_vars' con i corrispondenti 'operators' ai 'condition_values'.
    
    Parameters:
    df (pandas.DataFrame): Il dataframe da condizionare.
    condition_vars (list of str): Lista delle variabili su cui condizionare.
    operators (list of str): Lista degli operatori di confronto da usare ('>=', '<=', '>', '<', '==').
    condition_values (list): Lista dei valori (o limiti) corrispondenti per le variabili in 'condition_vars'.

    Returns:
    pandas.DataFrame: Un dataframe condizionato.
    """
    if not (isinstance(condition_vars, list) and isinstance(operators, list) and isinstance(condition_values, list)):
        raise ValueError("All parameters must be lists")
    if not (len(condition_vars) == len(operators) == len(condition_values)):
        raise ValueError("All input lists must have the same length")
    
    query_str = ' & '.join([f"{var} {op} @{var}_val" for var, op, var_val in zip(condition_vars, operators, condition_values)])
    return df.query(query_str)
#conditioned_df = condition_dataframe_on_ranges(df, ['X', 'Y'], ['>=', '<'], [2, 6])

# Example usage: Condition on X = 0 and Z = 0
df_ALPHA_Y0 = condition_on(df_ALPHA, 'Y', 1)
df_ALPHA_Z0 = condition_on(df_ALPHA, 'Z', 1)
#df_ALPHA_YZ0 = condition_dataframe_on_ranges(df_ALPHA, ['Y', 'Z'],  [])
df_BETA_Y0 = condition_on(df_BETA, 'Y', 1)
df_BETA_Z0 = condition_on(df_BETA, 'Z', 1)


corr_ALPHA_y0 = df_ALPHA_Y0.corr()
corr_ALPHA_z0 = df_ALPHA_Z0.corr()
corr_BETA_y0 = df_BETA_Y0.corr()
corr_BETA_z0 = df_BETA_Z0.corr()
print("\nCorrelation Matrix for ALPHA Model, Conditioning on Z:\n", corr_ALPHA_z0)
print("\nCorrelation Matrix for BETA Model, Conditioning on Z:\n", corr_BETA_z0)




# pairs = [('X', 'Y'), ('X', 'Z'), ('X', 'W'), ('Z', 'W'), ('Y', 'W'), ('Z', 'Y')]
# for x_var, y_var in pairs:

#     # # No conditiong plots
#     # sns.jointplot(data=df_ALPHA, x=x_var, y=y_var, kind='scatter', marginal_kws=dict(bins=30, fill=True))
#     # plt.title(f'Scatter plot of {x_var} vs {y_var} for ALPHA Model', pad=70)
#     # plt.show()

#     # For ALPHA Model conditioned on X = 0
#     sns.jointplot(data=df_ALPHA_X0, x=x_var, y=y_var, kind='scatter', marginal_kws=dict(bins=30, fill=True))
#     plt.title(f'ALPHA: {x_var} vs {y_var} | X ~ 0', pad=70)
#     plt.show()

#     # For ALPHA Model conditioned on Z = 0
#     sns.jointplot(data=df_ALPHA_Z0, x=x_var, y=y_var, kind='scatter', marginal_kws=dict(bins=30, fill=True))
#     plt.title(f'ALPHA: {x_var} vs {y_var} | Z ~ 0', pad=70)
#     plt.show()

#     # sns.jointplot(data=df_BETA, x=x_var, y=y_var, kind='scatter', marginal_kws=dict(bins=30, fill=True))
#     # plt.title(f'Scatter plot of {x_var} vs {y_var} for BETA Model', pad=70)
#     # plt.show()