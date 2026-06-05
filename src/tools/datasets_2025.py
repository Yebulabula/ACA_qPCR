
import pickle
import os
from pathlib import Path  
import pandas as pd
import numpy as np
import sys

dir_scr = os.path.join('src', 'tools')
if dir_scr not in sys.path:
    sys.path.append(dir_scr)

helper_path = os.path.join(dir_scr, 'DEEP_ACA_LK_functions_v2.py')
if os.path.exists(helper_path):
    with open(helper_path) as f:
        code = f.read()
    exec(code)

## experimental data

dir_data_2025 = '7_plex'
dir_out = '7_plex_output'

df_qPCR_GB_2025 = pd.read_csv(os.path.join(dir_data_2025, "df_qPCR_GB_2025.csv"))
df_qPCR_SP_2025 = pd.read_csv(os.path.join(dir_data_2025, "df_qPCR_SP_2025.csv"))
df_dPCR_GB_2025 = pd.read_csv(os.path.join(dir_data_2025, "df_dPCR_GB_2025.csv"))
df_dPCR_SP_2025 = pd.read_csv(os.path.join(dir_data_2025, "df_dPCR_SP_2025.csv"))
df_dPCR_SP_2025_clean = pd.read_csv(os.path.join(dir_data_2025, "df_dPCR_SP_2025_clean.csv"))
df_qPCR_GB_2025_conc3 = pd.read_csv(os.path.join(dir_data_2025, "df_qPCR_GB_2025_conc3.csv"))
df_qPCR_GB_2025_conc3_clean = pd.read_csv(os.path.join(dir_data_2025, "df_qPCR_GB_2025_conc3_clean.csv")).dropna()
df_simulated_ACs_norm = pd.read_csv(os.path.join(dir_data_2025, "df_simulated_ACs_norm.csv"))

## simulated data

"""
rows = []

for target, conc_dict in dict_simulated_ACs.items():
    for conc, fluo_values in conc_dict.items():
        row = [target] + fluo_values.tolist() + [conc]
        rows.append(row)

columns = ['Target'] + list(range(1, 61)) + ['Conc']

df_simulated_ACs = pd.DataFrame(rows, columns=columns)

df_simulated_ACs.loc[df_simulated_ACs.Target == "Ade", "Target"] = "Adeno"
df_simulated_ACs.loc[df_simulated_ACs.Target == "C22", "Target"] = "Cov_229E"
df_simulated_ACs.loc[df_simulated_ACs.Target == "CHK", "Target"] = "Cov_HKU1"
df_simulated_ACs.loc[df_simulated_ACs.Target == "CNL", "Target"] = "Cov_NL63"
df_simulated_ACs.loc[df_simulated_ACs.Target == "COC", "Target"] = "Cov_OC43"

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df_simulated_ACs.Target)
df_simulated_ACs['Target_cat'] = le.transform(df_simulated_ACs.Target)

df_simulated_ACs.to_csv(os.path.join(dir_data_2025, 'df_simulated_ACs.csv'), index=False)

## normalize values of simluated ACs
N_CYCLE_AVE=10
control=['Cov_OC43']
df_simulated_ACs.columns = df_simulated_ACs.columns.astype(str)
df_tmp = df_simulated_ACs.copy()
col_cycles = [x for x in df_tmp.columns if x.isdigit()]
df_ctrl = df_tmp[df_tmp.Target.isin(control)].copy()
df_ctrl['last_n_cycles_avg'] = (
    df_ctrl.filter(regex=r'\d+\.?\d*').iloc[:, -N_CYCLE_AVE:].astype(float).mean(axis=1)
)
ctrl_median = np.median(df_ctrl['last_n_cycles_avg'])
df_tmp.loc[:, col_cycles] = df_tmp.loc[:, col_cycles].divide(ctrl_median)

df_simulated_ACs_norm = df_tmp.copy()
df_simulated_ACs_norm.to_csv(os.path.join(dir_data_2025, 'df_simulated_ACs_norm.csv'), index=False)

"""

#######################################

plot_curves_2(df_dPCR_GB_2025, title = 'df_dPCR_GB_2025')
plot_curves_2(df_dPCR_SP_2025, title = 'df_dPCR_SP_2025')
plot_curves_2(df_dPCR_SP_2025_clean, title = 'df_dPCR_SP_2025_clean')

plot_curves_qPCR(df_qPCR_GB_2025, title = 'df_qPCR_GB_2025')
plot_curves_qPCR(df_qPCR_SP_2025, title = 'df_qPCR_SP_2025')
plot_curves_qPCR(df_qPCR_SP_2025_conc3, title = 'df_qPCR_SP_2025_conc3')
plot_curves_qPCR(df_qPCR_SP_2025_conc3_clean, title = 'df_qPCR_SP_2025_conc3_clean')
plot_curves_2(df_simulated_ACs_norm, title = 'df_simulated_ACs')


#######################################
## tmp_datasets

tmp_dataset_1 = pd.concat([df_qPCR_GB_2025, df_qPCR_SP_2025], axis = 0)
plot_curves_qPCR(tmp_dataset_1)
tmp_dataset_1.to_csv(os.path.join(dir_data_2025, 'tmp_dataset_1.csv'), index=False)

tmp_GB_1 = df_qPCR_GB_2025[df_qPCR_GB_2025.Conc > 6]
tmp_SP_1 = df_qPCR_SP_2025[df_qPCR_SP_2025.Conc > 6]
tmp_dataset_1 = pd.concat([tmp_GB_1, tmp_SP_1], axis = 0)
plot_curves_qPCR(tmp_dataset_1)
tmp_dataset_1.to_csv(os.path.join(dir_data_2025, 'tmp_dataset_1.csv'), index=False)

#######################################
## sample 
