"""
Created on 220115
run_stat_hyperparam.py
Script to produce simulated data and perform a hyperparameter search for the
LS-periodogram statistics.

@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import feat_eng as fe
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis
import pandas as pd
import itertools as it

# >> what to run?
hyperparam_opt = False
hp_opt_asassn = False
plot = True

# >> inputs
datapath     = '/scratch/data/tess/lcur/spoc/'
metapath     = '/scratch/data/tess/meta/'
timescale    = 1
savepath = '/scratch/echickle/timescale-'+str(timescale)+'sector/feat_eng/'
 # >> hyperparameter result txt
fname_mock = savepath+'lspm-feats-mock/lspm_stats_hyperparam.txt'

batch_size   = 32

# ------------------------------------------------------------------------------
# fe.spoc_noise(datapath, metapath, savepath)

thresh_max = np.arange(2, 30)
thresh_min = np.arange(0.5, 3, 0.5)
kernel = np.arange(3, 35, 2) # >> put into terms of df?
n_terms = np.arange(1,3)
hpsets = list(it.product(*[thresh_max, thresh_min, kernel, n_terms]))
np.random.shuffle(hpsets)
print('Number of hyperparameter combinations: '+str(len(hpsets)))

if hyperparam_opt:
    frequency = fe.get_frequency_grid('/scratch/echickle/', timescale)

    with open(fname_mock, 'w') as f:
        line = 'Hyperparam_Set,Light_Curve,thresh_max,thresh_min,kernel,'+\
               'n_terms,Npeaks_true,Npeaks_pred,P_true,P_pred,A_true,A_pred,'
        for i in ['P', 'Pratio', 'A', 'Aratio']:
            for j in ['mean', 'stdv', 'skew', 'kurt']:
                line += i+'_'+j+'_true,'
                line += i+'_'+j+'_pred,'
        f.write(line[:-1]+'\n') # >> remove last comma and add new line char


    hpset = 0
    for h1, h2, h3, h4 in hpsets:
        hpset += 1
        print('Hyperparameter Set: '+str(hpset))
        print('thresh_max: '+str(h1))
        print('thresh_min: '+str(h2))
        print('kernel: '+str(h3))
        print('n_terms: '+str(h4))

        t_batch, y_batch, period, amp = \
            fe.periodic_simulated_data(savepath+'lspm-feats-mock/', datapath,
                                       metapath, kernel=h3,
                                       timescale=1, freq=frequency,
                                       n_samples=batch_size)    


        for i in range(batch_size):
            true_feats = fe.calc_lspm_stats(period[i], amp[i])

            power = LombScargle(t_batch[i], y_batch[i], nterms=h4).power(frequency)    
            P_pred, A_pred, lspm_feats = \
            fe.peak_finder(t=t_batch[i], y=y_batch[i], frequency=frequency,
                           power=power, datapath=datapath, fmax=1/(1/24.),
                           thresh_max=h1, thresh_min=h2, n_terms=h4,
                           timescale=timescale,
                           ticid='hpset'+str(hpset)+'_lc'+str(i), 
                           savepath=savepath+'lspm-feats-mock/')
            # pdb.set_trace()
            line = ','.join(map(str, [hpset, i, h1, h2, h3, h4]))+','
            if len(P_pred) > 0:
                line += ','.join(map(str, [period[i][0], P_pred[0], amp[i][0],
                                           A_pred[0]]))+','
            else:
                line += ','.join(map(str, [period[i][0], 0, amp[i][0], 0]))+','

            for true, pred in zip(true_feats, lspm_feats):
                line += str(true)+','
                line += str(pred)+','
            with open(fname_mock, 'a') as f:
                f.write(line[:-1]+'\n') # >> remove last comma and add new line char

if hp_opt_asassn:
    frequency = fe.get_frequency_grid('/scratch/echickle/', timescale)

    with open(fname, 'w') as f:
        line = 'Hyperparam_Set,Light_Curve,thresh_max,thresh_min,kernel,'+\
               'n_terms,P_true,P_pred,A_true,A_pred\n'
        f.write(line) 

    hpset = 0
    for h1, h2, h3, h4 in hpsets:
        hpset += 1
        print('Hyperparameter Set: '+str(hpset))
        print('thresh_max: '+str(h1))
        print('thresh_min: '+str(h2))
        print('kernel: '+str(h3))
        print('n_terms: '+str(h4))

        t_batch, y_batch, period, amp = \
            fe.periodic_simulated_data(savepath, datapath, metapath,
                                       timescale=1, freq=frequency,
                                       n_samples=batch_size)    


        for i in range(batch_size):
            true_feats = fe.calc_lspm_stats(period[i], amp[i])

            power = LombScargle(t_batch[i], y_batch[i], nterms=h4).power(frequency)    
            P_pred, A_pred, lspm_feats = fe.peak_finder(t=t_batch[i], y=y_batch[i],
                                                  frequency=frequency,
                                                  power=power, datapath=datapath,
                                                  ticid='hpset'+str(hpset)+'_lc'+str(i),
                                                  timescale=timescale,
                                                  savepath=savepath)
            # >> Hyperparam_Set,Light_Curve,
            line = ','.join(map(str, [hpset, i, h1, h2, h3, h4]))+','
            line += ','.join(map(str, [period[i][0], P_pred[0], amp[i][0],
                                       A_pred[0]]))+','
            for true, pred in zip(true_feats, lspm_feats):
                line += str(true)+','
                line += str(pred)+','
            with open(fname, 'a') as f:
                f.write(line[:-1]+'\n') # >> remove last comma and add new line char


if plot:
    # >> Create correlation matrix
    with open(fname_mock, 'r') as f:
        cols = f.readline()[:-1].split(',')
    data = np.loadtxt(fname_mock, skiprows=1, delimiter=',')
    err = [] # >> mean square error
    new_cols = cols[2:6]
    for i in range(int((len(cols)-6)/2)): # >> loop through features (P, Pratio, etc.)
        ind = 6+(i*2)
        err.append(np.abs(data[:,ind+1]-data[:,ind]))
        new_cols.append('_'.join(cols[ind].split('_')[:-1])+'_err')
    cols = new_cols
    data = np.concatenate([data[:,2:6], np.array(err).T], axis=1)

    pt.plot_corr_matrix(data, savepath+'lspm-feats-mock/', cols=cols,
                        annot_kws={'size': 'xx-small'})

    best_hpset = hpsets[np.argmin(data[:,np.nonzero(np.array(cols)==\
                                                    'P_skew_err')[0][0]])]
    print('Best hyperparameter set :'+str(best_hpset))

