import numpy as np
import pandas as pd
import scipy.optimize as opt
import os
import multiprocessing
import matplotlib.pyplot as plt

# Directories
dir_data_2025 = '7_plex'
dir_out = '7_plex_output'

# Load data
df = pd.read_csv(os.path.join(dir_data_2025, 'df_dPCR_SP_2025.csv'))  # Adjust filename if needed
df = df.filter(regex=r'\d+\.?\d*')  # Keep only numerical columns
curves = df.to_numpy()

# defines the 5 parameter sigmoid model (universal notations)
def sigmoid5_un(x, Fm, Fb, Sc, Cs, As):
    return Fm / (1. + np.exp(-(x-Cs)*Sc))**As + Fb

def sigmoid6_un(x, Fm, Fb, Sc, Cs, As, Bs):
    return Fm / ((1. + np.exp(-(x - Cs) * Sc))**As)**Bs + Fb

# Define residual function
def res_fun(param, x, y_obs):
    return sigmoid5_un(x, *param) - y_obs

# Define a LS loss function
def Loss(param, x, y_obs):
    residual = res_fun(param, x, y_obs)
    return 0.5*np.sum(residual**2)

def fitDF_5(curves, start, param_bounds, sigmoid):
    
    # creates a new dataframe to store generated parameters from each curve 
    param_df = pd.DataFrame(columns=['Fm', 'Fb', 'Sc', 'Cs', 'As', 'mse'])
    x = np.arange(1, 61, 1)

    # looping through every single amplification curve 
    for i in range(curves):
        y = curves[i]
        
        # fits each amplification curve to the sigmoid function
        try:
            popt, pcov = opt.curve_fit(sigmoid, x, y, p0=start, bounds=param_bounds, method='trf')
        except:
            print('Cannot find optimal solutions for curve # '+ str(i))
            popt = param_bounds[1]
        mean_square_error = np.mean((y-sigmoid(x, *popt))**2)
        param_df.loc[i] = [popt[0], popt[1], popt[2], popt[3], popt[4], mean_square_error]

    return param_df

# def fitDF_gb(base_df, NMETA, start, param_bounds):
    
#     current_df = base_df.copy()
    
#     x = current_df.iloc[:,NMETA:].T.index.astype(float)
#     sample_num = current_df.shape[0]

#     # creates a new dataframe to store generated parameters from each curve 
#     param_df = pd.DataFrame(columns=['Fm', 'Fb', 'Sc', 'Cs', 'As', 'mse'])

#     # looping through every single amplification curve 
#     for i in range(sample_num):
#         y = current_df.iloc[i, NMETA:].T.copy()
        
#         # fits each amplification curve to the sigmoid function
#         result_obj = opt.dual_annealing(Loss, bounds=np.array(param_bounds).T, args=(np.array(x), np.array(y)) )
#         popt = result_obj.x
#         mean_square_error = np.mean((y-sigmoid5_un(x, *popt))**2)
#         param_df.loc[i] = [popt[0], popt[1], popt[2], popt[3], popt[4], mean_square_error]
    

#     param_df.index = base_df.index #update indexes
#     param_df_with_meta = pd.concat([base_df.iloc[:,:NMETA],param_df],axis=1)

#     return param_df_with_meta


#%% FITTING - INITIAL - 5 PARAMETERS
##########################################################################

## Split dataframe into n for parallel computing
n_jobs = multiprocessing.cpu_count()
split_curves = np.array_split(curves.sample(frac=0.01, random_state=9), n_jobs)

## Fit
results_obj = []
start_5 = (1, 0, 0.5, 20, 1) #p0 for Fm, Fb, Sc, Cs, As
param_bounds_5 = ((0, -1, 0, 0, 0), (200, 1, 2, 50, 10))
if __name__ == '__main__':
    __spec__ = None
    pool = multiprocessing.Pool()
    for df_ in split_df:
        results_obj.append(pool.apply_async(fitDF_5, (df_, start_5, param_bounds_5, sigmoid5_un)))
    print("------start----")
    pool.close()  
    pool.join() 
    print("-------end------")

## Collect everything
sample_list = []
for obj in results_obj:
    sample_list.append(obj.get())
sample_df_param_5 = pd.concat(sample_list)

## Save df
sample_df_param_5.to_csv(os.path.join(dir_out, 'sample_df_param_5.csv'))

## Inspect visually
param_set = ['Fm', 'Fb', 'Sc', 'Cs', 'As']
df_tmp = pd.concat([df.loc[sample_df_param_5.index], sample_df_param_5[param_set]], axis = 1)
fig, ax = plt.subplots(3,3,figsize = (8,6), dpi = 300)
ax = ax.flatten()
for i in np.arange(len(df_tmp.index[0:9])):
    ax[i].plot(df_tmp.iloc[i, NMETA: NMETA+30], c = 'black', label = df_tmp.Target.iloc[i])
    Fm = df_tmp.iloc[i,:]['Fm']
    Fb = df_tmp.iloc[i,:]['Fb']
    Sc = df_tmp.iloc[i,:]['Sc']
    Cs = df_tmp.iloc[i,:]['Cs']
    As = df_tmp.iloc[i,:]['As']
    ax[i].plot(sigmoid5_un(np.arange(1,31,1), Fm, Fb, Sc, Cs, As), c = 'red', label = 'fit')
    ax[i].set_xlabel('Cycle', size = 8)
    ax[i].set_ylabel('Fluorescence', size = 8)
    ax[i].set_xticks(np.arange(0, 35, 5)-1, size = 8)
    ax[i].set_yticks(np.arange(0, 70, 10), size = 8)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 8)
    handles, labels = ax[i].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[i].legend(by_label.values(), by_label.keys(), loc = 'upper left', fontsize = 7, title = [])
plt.tight_layout()
plt.show()

## Plot mse's
fig = plt.figure()
ax = fig.add_subplot(111)
plt.hist(sample_df_param_5.mse, bins = 100)


#%% FITTING - WHOLE DATA SET - 5 PARAMETERS
##########################################################################

## Split dataframe into n for parallel computing
n_jobs = multiprocessing.cpu_count()
split_df = np.array_split(df, n_jobs)

results_obj = []
param_set = param_set = ['Fm', 'Fb', 'Sc', 'Cs', 'As']
start_5 = np.array(sample_df_param_5[param_set].median()) 
param_bounds_5 = ((0, -5, 0, 5,0), (100, 5, 4, 30,10))
if __name__ == '__main__':
    __spec__ = None
    pool = multiprocessing.Pool()
    for df_ in split_df:
        results_obj.append(pool.apply_async(fitting.fitDF_5, (df_, NMETA, start_5, param_bounds_5, fitting.sigmoid5_un)))
    print("------start----")
    pool.close()  
    pool.join() 
    print("-------end------")

## Collect everything
fitted_list = []
for obj in results_obj:
    fitted_list.append(obj.get())
param_df_5 = pd.concat(fitted_list)

## Save df
param_df_5.to_csv(os.path.join(dir_out, 'param_df_5.csv'))

## Inspect visually
param_set = ['Fm', 'Fb', 'Sc', 'Cs', 'As']
df_tmp = pd.concat([df.loc[param_df_5.index], param_df_5[param_set]], axis = 1)
fig, ax = plt.subplots(3,3,figsize = (8,6), dpi = 300)
ax = ax.flatten()
for i in np.arange(len(df_tmp.index[0:9])):
    ax[i].plot(df_tmp.iloc[i, NMETA: NMETA+30], c = 'black', label = df_tmp.Target.iloc[i])
    Fm = df_tmp.iloc[i,:]['Fm']
    Fb = df_tmp.iloc[i,:]['Fb']
    Sc = df_tmp.iloc[i,:]['Sc']
    Cs = df_tmp.iloc[i,:]['Cs']
    As = df_tmp.iloc[i,:]['As']
    ax[i].plot(fitting.sigmoid5_un(np.arange(1,31,1), Fm, Fb, Sc, Cs, As), c = 'red', label = 'fit')
    ax[i].set_xlabel('Cycle', size = 8)
    ax[i].set_ylabel('Fluorescence', size = 8)
    ax[i].set_xticks(np.arange(0, 35, 5)-1, size = 8)
    ax[i].set_yticks(np.arange(0, 70, 10), size = 8)
    ax[i].tick_params(axis = 'both', which = 'major', labelsize = 8)
    handles, labels = ax[i].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[i].legend(by_label.values(), by_label.keys(), loc = 'upper left', fontsize = 7, title = [])
plt.tight_layout()
plt.show()

## Plot mse's
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.hist(param_df_4.mse, bins = 100)
ax.hist(param_df_5.mse, bins = 100, range = (0,5))
ax.set_title("Histogram of MSEs")
plt.tight_layout()
plt.show()
