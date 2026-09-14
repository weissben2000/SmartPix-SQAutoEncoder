import os, subprocess
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from training_metadata import *
import json


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import warnings
warnings.filterwarnings("ignore")

maxval=1e9
minval=1e-9
pi = 3.14159265359

colors = list(mcolors.TABLEAU_COLORS.keys())
markers = ['o','s','v','^']

fontsize=16
plt.rc('font', size=fontsize-4)

with open('training_dict_new.json', 'r') as file:
    training_dict = json.load(file)

def grep(string, filename):

    out_list = []

    cmd = "grep '" + string + "' " + filename
    #print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Get output data
    cmd_out = result.stdout.splitlines()
    for line in cmd_out:
        #print(line)

        if '#' in line:
            print("Skipping line " + line)
            continue
        out_list += [line.split(':')[-1].strip()]

    return out_list

'''Training history'''
def training_history(ax, weights_path, loss_name = 'CCE loss'):
    training_cp_path = os.path.join(weights_path, 'training_log.csv')
    training_history = pd.read_csv(training_cp_path)
    
    ax.scatter(training_history['epoch'], training_history['loss'], s=3)
    ax.scatter(training_history['epoch'], training_history['val_loss'], s=3)
    ax.legend(['training', 'validation'])
    ax.grid(True)
    ax.set_xlabel('Epochs', fontsize = 14)
    ax.set_title(loss_name, fontsize = 16)
    ax.set_ylabel(loss_name, fontsize = 14)

    

'''Threshold migration and loss optimization for one training'''
def threshold_migration(ax, weights_path, FE_bits = 2, target = None):
    Nth = 2**FE_bits-1
    SQ_log = pd.read_csv(os.path.join(weights_path, 'soft_quantizer_state_log.csv'))
    th_keys = [f'threshold_{i}' for i in range(Nth)]
    for i, th_key in enumerate(th_keys):
        epochs = SQ_log['epoch']
        th_charge = SQ_log[th_key]
        ax.scatter(epochs, th_charge, s=3, label = f'Th{i}')
    
    ax.grid()
    ax.legend()
    ax.set_xlabel('Epochs', fontsize =14)
    ax.set_ylabel('Charge threshold [e]', fontsize = 14)
    if target is not None:
        target_text = f'\n target={target}'
    else:
        target_text = ''
    ax.set_title(f'{FE_bits}bit SQ_DAE thresholds '+target_text, fontsize = 16)

'''Residual construction'''
def rescale(df, scale_factor):

    # Only rescales means and sigmas, not correlations
    for v in ['x','y','cotA','cotB']:
        if v in df.columns:
            df[v] = scale_factor[v]*df[v]
            df[v+'true'] = scale_factor[v]*df[v+'true']
        if 'sigma'+v in df.columns:
            df['sigma'+v] = scale_factor[v]*df['sigma'+v]

    return df

def construct_df(model, test_generator, scale_factor):
    maxval=1e9
    minval=1e-9
    # predicts test data
    p_test = model.predict(test_generator)

    # Jennet: TBH not sure why this is needed but ok
    complete_truth = None
    for _, y in tqdm(test_generator):
        if complete_truth is None:
            complete_truth = y
        else:
            complete_truth = np.concatenate((complete_truth, y), axis=0)

    # creates df with all predicted values and matrix elements - 4 predictions, all 10 unique matrix elements
    df = pd.DataFrame(p_test,columns=['x','M11','y','M22','cotA','M33','cotB','M44',
                                      'M21','M31','M32','M41','M42','M43'])

    # stores all true values in same matrix as xtrue, ytrue, etc.
    df['xtrue'] = complete_truth[:,0]
    print(np.mean(df['xtrue']),np.min(df['xtrue']),np.max(df['xtrue']))
    df['ytrue'] = complete_truth[:,1]
    df['cotAtrue'] = complete_truth[:,2]
    df['cotBtrue'] = complete_truth[:,3]
    df['M11'] = minval+tf.math.maximum(df['M11'], 0)
    df['M22'] = minval+tf.math.maximum(df['M22'], 0)
    df['M33'] = minval+tf.math.maximum(df['M33'], 0)
    df['M44'] = minval+tf.math.maximum(df['M44'], 0)

    df['sigmax'] = abs(df['M11'])
    df['sigmay'] = np.sqrt(df['M21']**2 + df['M22']**2)
    df['sigmacotA'] = np.sqrt(df['M31']**2+df['M32']**2+df['M33']**2)
    df['sigmacotB'] = np.sqrt(df['M41']**2+df['M42']**2+df['M43']**2+df['M44']**2)

    extra_columns = ['M11','M22','M33','M44','M21','M31','M32', 'M41', 'M42', 'M43']
    df.drop(labels=extra_columns, axis=1, inplace=True)

    df = rescale(df, scale_factor)

    # calculates residuals for x, y, cotA, cotB
    for v in ['x','y','cotA','cotB']:
        if v in df.columns:
            df['residual'+v] = (df[v] - df[v+'true'])
            if 'sigma'+v in df.columns:
                df['pull'+v] = (df[v] - df[v+'true'])/df['sigma'+v]

    for v in ['cotA','cotB']:
        if v in df.columns:
            df[v[-1]] = inverse_cot(df[v].values)*180/np.pi
            df[v[-1]+'true'] = inverse_cot(df[v+'true'].values)*180/np.pi
            df['residual'+v[-1]] = (df[v[-1]] - df[v[-1]+'true'])

    return df

def construct_df_slim(model, test_generator, scale_factor):
    maxval=1e9
    minval=1e-9
    # predicts test data
    p_test = model.predict(test_generator)

    # Jennet: TBH not sure why this is needed but ok
    complete_truth = None
    for _, y in tqdm(test_generator):
        if complete_truth is None:
            complete_truth = y
        else:
            complete_truth = np.concatenate((complete_truth, y), axis=0)

    # creates df with all predicted values and matrix elements - 4 predictions, all 10 unique matrix elements
    df = pd.DataFrame(p_test,columns=['x','y','cotB'])

    # stores all true values in same matrix as xtrue, ytrue, etc.
    df['xtrue'] = complete_truth[:,0]
    print(np.mean(df['xtrue']),np.min(df['xtrue']),np.max(df['xtrue']))
    df['ytrue'] = complete_truth[:,1]
    df['cotBtrue'] = complete_truth[:,2]

    df = rescale(df, scale_factor)

    # calculates residuals for x, y, cotB
    for v in ['x','y','cotB']:
        if v in df.columns:
            df['residual'+v] = (df[v] - df[v+'true'])
            if 'sigma'+v in df.columns:
                df['pull'+v] = (df[v] - df[v+'true'])/df['sigma'+v]

    for v in ['cotB']:
        if v in df.columns:
            df[v[-1]] = inverse_cot(df[v].values)*180/np.pi
            df[v[-1]+'true'] = inverse_cot(df[v+'true'].values)*180/np.pi
            df['residual'+v[-1]] = (df[v[-1]] - df[v[-1]+'true'])

    return df

pred_cols = [str(x)+'_pred' for x in range(16*16*2)]
true_cols = [str(x)+'_true' for x in range(16*16*2)]
def construct_df_SQAutoEncoder(model, x, y):
    '''Constructs paired prediction and truth values from DAE_DG x, y for one batch'''
    
    this_df = pd.DataFrame()

    pred_cols = [str(x)+'_pred' for i in range(16*16*2)]
    true_cols = [str(x)+'_true' for i in range(16*16*2)]

    batch_size = len(y)
    
    truth = np.argmax(y, axis = -1).reshape(batch_size, 16*16*2)
    # print(truth.shape)
    predictions = model.predict(x, verbose = 0)
    predictions = np.argmax(predictions, axis = -1).reshape(batch_size, 16*16*2)
    # # print(predictions.shape)
    
    this_df[pred_cols] = predictions
    this_df[true_cols] = truth
    # print(len(pred_df))
    return this_df

def clusters_from_df(df):
    l = len(df)
    pred_cols = [str(x)+'_pred' for i in range(16*16*2)]
    true_cols = [str(x)+'_true' for i in range(16*16*2)]
    
    pred_clusters = df[pred_cols].values.reshape(l,16,16,2)
    true_clusters = df[true_cols].values.reshape(l,16,16,2)
    return pred_clusters, true_clusters

    
'''Residuals: 1D'''
def shortest_interval_68(data, center_type='mean'):
    """
    Compute the shortest interval that contains 68% of the data.
    """
    data = np.sort(data)
    n = len(data)
    ci_size = int(np.floor(0.68 * n))
    min_width = float("inf")
    min_i = 0

    for i in range(n - ci_size):
        width = data[i + ci_size] - data[i]
        if width < min_width:
            min_width = width
            min_i = i

    low = data[min_i]
    high = data[min_i + ci_size]

    if center_type == 'mean':
        center = np.mean(data)
    else:
        center = np.median(data)

    return {
        "center": center,
        "error_low": float(center - low),
        "error_high": float(high - center),
        "center_type": center_type
    }

def draw_one_vbl(vbl, x, d, ax_ii, vbl_names):
    
    x_array = [-1*x+0.3]
    y_array = [d['mean_'+vbl]]
    yerr_array = [[d['down68_'+vbl]],[d['up68_'+vbl]]]
    yerr2_array = [[d['mean_downsigma'+vbl]],[d['mean_upsigma'+vbl]]]
    
    
    line1 = ax_ii.errorbar(x = y_array, y = x_array, xerr=yerr_array, color = colors[x], linestyle='')
    line2 = ax_ii.errorbar(x = y_array, y = x_array, xerr=yerr2_array, elinewidth=10, alpha=0.2, color = colors[x], linestyle='')
    
    dot1 = ax_ii.scatter(y_array, x_array, marker='o', color = colors[x])
    
    ax_ii.set_xlabel(vbl_names[vbl],fontsize=fontsize)
    ax_ii.set_yticks([])

    return [line1,line2], [dot1]
    
def draw_one_model(x, d, names, ax, vbl_names, slim = False):

    if slim:     
        line, dots = draw_one_vbl_slim('x', x, d, ax[1][0], vbl_names)
        line, dots = draw_one_vbl_slim('y', x, d, ax[1][1], vbl_names)
        line, dots = draw_one_vbl_slim('B', x, d, ax[1][2], vbl_names)
        
    else:
        line, dots = draw_one_vbl('x', x, d, ax[1][0], vbl_names)
        line, dots = draw_one_vbl('y', x, d, ax[1][1], vbl_names)
        line, dots = draw_one_vbl('A', x, d, ax[1][2], vbl_names)
        line, dots = draw_one_vbl('B', x, d, ax[1][3], vbl_names)

    return line, dots

'''Categorization accuracy plots'''
def PCatAccuarcyByLevel(X_val, model = None, QThresh = None):
    cshape = (16,16,2)
    npixels = cshape[0]*cshape[1]
    
    acc_all = pd.DataFrame()

    desc_string = "supplied model" if model is not None else f'HQ Ths: [{QThresh}]'
    for i in tqdm(range(len(X_val)), desc = f'Computing accuracy with {desc_string}'):
        true_clusters = np.argmax(X_val[i][1][:][:,:,:],axis = -1)
        test_clusters = X_val[i][0][:][:,:,:].numpy()
        
        if model is not None:
            p_out = np.argmax(model.predict(test_clusters,  verbose = 0), axis=-1)
            sumax = (1,2,3)
        elif QThresh is not None:
            true_clusters = true_clusters.reshape(true_clusters.shape[0], math.prod(cshape))
            test_clusters = test_clusters.reshape(true_clusters.shape[0], math.prod(cshape))
            p_out = quantize_manual(test_clusters, charge_levels = QThresh, quant_values=np.arange(len(QThresh)+1) )
            sumax = 1
            # print(p_out.shape, true_clusters.shape)
        acc= pd.DataFrame()
        levels = np.arange(np.max(true_clusters)+1)
        for l in levels:
            true_catmask = true_clusters == l
            pred_catmask = p_out == l
            
            npix_perlevel = np.sum(true_catmask, axis = sumax) 
            correct_preds = np.sum((true_catmask & pred_catmask), axis = sumax) 
            # print("npix and correct preds shape: ", npix_perlevel.shape, correct_preds.shape)
            acc[str(l)] = correct_preds/npix_perlevel
            # print(acc.shape)
        acc_all = pd.concat([acc_all, acc])
        
    return acc_all

def PCatAccuarcyByTS(X_val, model = None, QThresh = None):
    cshape = (16,16,2)
    npixels = cshape[0]*cshape[1]
    
    ts0acc = np.array([])
    ts19acc = np.array([])
        
    for i in tqdm(range(len(validation_generator)), desc = 'Computing accuracy with supplied model'):
        true_clusters = np.argmax(X_val[i][1][:][:,:,:,:],axis = -1)
        test_clusters = X_val[i][0][:][:,:,:]
        
        if model is not None:
            p_out = np.argmax(model.predict(test_clusters,  verbose = 0), axis=-1)
            ts0acc = np.append(ts0acc,(npixels - np.count_nonzero(p_out[:,:,:,0]-true_clusters[:,:,:,0],  axis = (1,2)))/npixels)
            ts19acc = np.append(ts19acc,(npixels - np.count_nonzero(p_out[:,:,:,1]-true_clusters[:,:,:,1],  axis = (1,2)))/npixels)

        elif QThresh is not None:
            true_clusters_TS0 = true_clusters[:,:,:,0]
            test_clusters_TS0 = test_clusters[:,:,:,0].numpy()
            true_clusters_TS19 = true_clusters[:,:,:,-1]
            test_clusters_TS19 = test_clusters[:,:,:,-1].numpy()
    
            true_clusters_TS0 = true_clusters_TS0.reshape(true_clusters_TS0.shape[0], npixels)
            true_clusters_TS19 = true_clusters_TS19.reshape(true_clusters_TS19.shape[0], npixels)
            test_clusters_TS0 = test_clusters_TS0.reshape(true_clusters_TS0.shape[0], npixels)
            test_clusters_TS19 = test_clusters_TS19.reshape(true_clusters_TS19.shape[0], npixels)
            
            p_out_TS0 = quantize_manual(test_clusters_TS0, charge_levels = QThresh, quant_values=[0,1,2,3])
            p_out_TS19 = quantize_manual(test_clusters_TS19, charge_levels = QThresh, quant_values=[0,1,2,3])
            ts0acc = np.append(ts0acc,(npixels - np.count_nonzero(p_out_TS0-true_clusters_TS0,  axis = 1))/npixels)
            ts19acc = np.append(ts19acc,(npixels - np.count_nonzero(p_out_TS19-true_clusters_TS19,  axis = 1))/npixels)
    return ts0acc, ts19acc

'''Residuals: var binned'''
from scipy.optimize import curve_fit

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def pull_plot(ax, var, name, label, p=0):
    
    h = ax.hist(df[var],bins=np.linspace(-5,5,50),histtype='step', label=label)
    ax.set_xlabel(name)
    ax.set_yscale('log')

    ydata = h[0]
    xdata = h[1][:-1]+3/50.

    pars, cov = curve_fit(gauss,xdata,ydata)

    xbins = np.linspace(-5,5,100)
    ax.plot(xbins,gauss(xbins,pars[0],pars[1],pars[2]),color='black')
    ax.set_ylim(0.5,100000)

    print('Mean',pars[1])
    print('Sigma',pars[2])

    if p:
        ax.text(-1,25,"$\mu$="+str(round(pars[1],2)))
        ax.text(-1,10,"$\sigma$="+str(round(abs(pars[2]),2)))

def residual_plot(ax, thisdf, var1, var2, name, label, scaling=1.0,color = 'blue'):
    
    nbins = 15
    
    var1_scaled = thisdf[var1] * scaling
    var2_scaled = thisdf[var2] * scaling
    residual_scaled = var1_scaled - var2_scaled
    
    xmin = np.min(var1_scaled)
    xmax = np.max(var1_scaled)
    
    step = 1.0*(xmax-xmin)/nbins
    
    x = sns.regplot(x=var1_scaled, y=residual_scaled, x_bins=np.linspace(xmin,xmax,nbins), fit_reg=None, marker='.', ax=ax, label=label, color = color)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    
    thisdf['residual'+var2] = residual_scaled
    # print(var1)
    
    means = []
    upbar = []
    downbar = []
    for i in range(0,nbins):
        means += [np.mean(thisdf['residual'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)])]
        upbar += [means[i] + np.mean(thisdf['sigma'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)] * scaling)]
        downbar += [means[i] - np.mean(thisdf['sigma'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)] * scaling)]
    ax.fill_between(x=np.linspace(xmin,xmax,nbins),y1=upbar,y2=downbar, alpha=0.2, color = color)
    print(f"{name} spread: ",np.mean(np.array(upbar)-np.array(downbar)))
    
def inverse_cot(cota):
    a = np.arctan(1.0/cota)
    a[np.where(a<0)] = a[np.where(a<0)] + pi
    return a    

def residual_plot_deg(ax, thisdf, var1, var2, name, label, scaling=1.0, color = 'blue'):
    # positions
    if 'cot' not in var1:
        residual_plot(ax, thisdf, var1, var2, name, label=label, scaling=scaling)
        return

    thisdf['angle'] = inverse_cot(thisdf[var2].values * scaling)*180/pi
    
    thisdf['angleup'] = abs(inverse_cot((thisdf[var2].values + thisdf['sigma'+var2].values) * scaling)*180/pi - thisdf['angle'])
    thisdf['angledown'] = abs(inverse_cot((thisdf[var2].values - thisdf['sigma'+var2].values) * scaling)*180/pi - thisdf['angle'])
    thisdf['angletrue'] = inverse_cot(thisdf[var1].values * scaling)*180/pi
        
    var1 = 'angletrue'
    var2 = 'angle'
    
    nbins = 15
    xmin = np.min(thisdf[var1])
    xmax = np.max(thisdf[var1])
    
    step = 1.0*(xmax-xmin)/nbins
        
    x = sns.regplot(x=thisdf[var1], y=(thisdf[var1]-thisdf[var2]), x_bins=np.linspace(xmin,xmax,nbins), fit_reg=None, marker='.', ax=ax, label=label, color = color)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    
    thisdf['residual'+var2] = (thisdf[var1]-thisdf[var2])
    # print(var1)
    
    means = []    
    upbar = []
    downbar = []
    for i in range(0,nbins):
        means += [np.mean(thisdf['residual'+var2][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
        upbar += [means[i] + np.mean(thisdf['angleup'][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
        downbar += [means[i] - np.mean(thisdf['angledown'][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
    #ax.scatter(x=np.linspace(xmin,xmax,nbins),y=means)
    ax.fill_between(x=np.linspace(xmin,xmax,nbins),y1=upbar,y2=downbar, alpha=0.2, color = color)
    print(f"{name} spread: ",np.mean(np.array(upbar)-np.array(downbar)))

#Create a table of residual widths for a set of models 
def compute_ResWidths(predictions, model_out = 'max'):
    residuals = {}
    # Mean predicted uncertainty 
    residuals['mean_upsigmax'] = float(np.mean(predictions['sigmax']))
    residuals['mean_downsigmax'] = float(np.mean(predictions['sigmax']))
    residuals['mean_upsigmay'] = float(np.mean(predictions['sigmay']))
    residuals['mean_downsigmay'] = float(np.mean(predictions['sigmay']))
               
    residuals['mean_upsigmacotB'] = float(np.mean(predictions['sigmacotB']))
    residuals['mean_downsigmacotB'] = float(np.mean(predictions['sigmacotB']))
    residuals['mean_upsigmaB'] = float(np.mean(abs(inverse_cot(predictions['cotB'].values + predictions['sigmacotB'].values)*180/np.pi - predictions['B'].values)))
    residuals['mean_downsigmaB'] = float(np.mean(abs(inverse_cot(predictions['cotB'].values - predictions['sigmacotB'].values)*180/np.pi - predictions['B'].values)))

    # Residuals for x and y (microns)
    for v in ['x','y']:
        # Mean
        residuals['mean_'+v] = float(np.mean(predictions['residual'+v]))
        # Std -- replace with smallest window containing 68%
        residuals['std_'+v] = float(np.std(predictions['residual'+v]))

        interval68 = shortest_interval_68(predictions['residual'+v], center_type='mean')
        residuals['up68_'+v] = interval68['error_high']
        residuals['down68_'+v] = interval68['error_low']

    # Residuals for cotB
    residuals['mean_cotB'] = float(np.mean(predictions['residualcotB']))
    residuals['std_cotB'] = float(np.std(predictions['residualcotB']))
    interval68_cotB = shortest_interval_68(predictions['residualcotB'], center_type='mean')
    residuals['up68_cotB'] = interval68_cotB['error_high']
    residuals['down68_cotB'] = interval68_cotB['error_low']

    # Residuals for B
    residuals['mean_B'] = float(np.mean(predictions['residualB']))
    residuals['std_B'] = float(np.std(predictions['residualB']))
    interval68_B = shortest_interval_68(predictions['residualB'], center_type='mean')
    residuals['up68_B'] = interval68_B['error_high']
    residuals['down68_B'] = interval68_B['error_low']
    
    if model_out != 'slim':
        residuals['mean_cotA'] = float(np.mean(predictions['residualcotA']))
        residuals['std_cotA'] = float(np.std(predictions['residualcotA']))
        interval68_cotA = shortest_interval_68(predictions['residualcotA'], center_type='mean')
        residuals['up68_cotA'] = interval68_cotA['error_high']
        residuals['down68_cotA'] = interval68_cotA['error_low']
    
        # residuals for A
        residuals['mean_A'] = float(np.mean(predictions['residualA']))
        residuals['std_A'] = float(np.std(predictions['residualA']))
        interval68_A = shortest_interval_68(predictions['residualA'], center_type='mean')
        residuals['up68_A'] = interval68_A['error_high']
        residuals['down68_A'] = interval68_A['error_low']
        residuals['mean_upsigmacotA'] = float(np.mean(predictions['sigmacotA']))
        residuals['mean_downsigmacotA'] = float(np.mean(predictions['sigmacotA']))
        residuals['mean_upsigmaA'] = float(np.mean(abs(inverse_cot(predictions['cotA'].values + predictions['sigmacotA'].values)*180/np.pi - predictions['A'].values)))
        residuals['mean_downsigmaA'] = float(np.mean(abs(inverse_cot(predictions['cotA'].values - predictions['sigmacotA'].values)*180/np.pi - predictions['A'].values))) 

    return residuals

def get_ModelResWidths_byNoise(model_list, model_out = 'max'):
    # with open('SQ_DAE_regression/residuals/residuals.json', 'r') as file:
    #     residuals = json.load(file)
    if model_out == 'max':
        var_list = ['x', 'y', 'A', 'B']
    elif model_out == 'slim':
        var_list = ['x', 'y', 'B']
    alldata = None
    noise = []
    res_width = []
    for m in model_list:
        this_data = pd.DataFrame()
        print(m)
        if m in training_dict.keys():
            this_data['noise']=[training_dict[m]['noise']]
            df = pd.read_parquet(f'./SQ_DAE_regression/residuals/{m}_residuals.parquet')
            residuals = compute_ResWidths(df, model_out = model_out)
            for var in var_list:
                vrw = residuals[f'up68_{var}'] + residuals[f'down68_{var}'] #plus bc up/down are both stored as positive
                this_data[f'{var}_width'] = [vrw]
            # print(this_data)
            if alldata is None:
                alldata = this_data
            else:
                alldata = pd.concat([alldata, this_data], axis = 0)
        else:
            raise ValueError(m, f' is not in {training_dict.keys()}')
    return alldata

    