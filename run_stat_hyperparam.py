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
from astropy.io import fits
from scipy.stats import skew, kurtosis
import pandas as pd
import itertools as it

# >> what to run?
hyperparam_opt = False
hp_opt_asassn = False
plot = False
plot_asassn = True

# >> inputs
datapath     = '/scratch/data/tess/lcur/spoc/'
metapath     = '/scratch/data/tess/meta/'
timescale    = 1
savepath = '/scratch/echickle/timescale-'+str(timescale)+'sector/feat_eng/'
 # >> hyperparameter result txt
fname_mock = savepath+'lspm-feats-mock/lspm_stats_hyperparam.txt'
fname_asas = savepath+'lspm-feats-mock/asas_sn_hyperparam.txt'
min_err = 'P_skew_err'


batch_size   = 8

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
    df = frequency[1] - frequency[0]

    with open(fname_mock, 'w') as f:
        line = 'Hyperparam_Set,Light_Curve,mag,precision,recall,'+\
               'thresh_max,thresh_min,kernel,'+\
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

        # -- produce simulated data --------------------------------------------
        t_batch, y_batch, period, amp, mag = \
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


            # -- compute recall, precision ------------------------------------
            precision, recall = fe.eval_simulated_data(P_pred, period[i], df)

            # -- save results --------------------------------------------------
            line = ','.join(map(str, [hpset, i, mag[i], precision, recall, h1,
                                      h2, h3, h4]))+','
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

    if os.path.exists(savepath+'lspm-feats-mock/asas_sn_period_amplitude.txt'):
        data = np.loadtxt(savepath+'lspm-feats-mock/asas_sn_period_amplitude.txt',
                          dtype='str', delimiter=',')
        ticid = data[:,0].astype('int') 
        otype, asasid = data[:,1], data[:,2]
        amplitude, period = data[:,3].astype('float'), data[:,4].astype('float')
    else:

        # -- get TICID cross-matched with ASAS-SN catalog ----------------------
        ticid = []
        otype = []
        asasid = []
        for i in range(1,27):
            print('Loading sector '+str(i))
            filo = np.loadtxt(metapath+'spoc/cat/sector-%02d'%i+'_asassn_cln.txt',
                              delimiter=',', dtype='str', skiprows=1)
            inds = np.nonzero(filo[:,1] != 'NONE')[0]
            for ind in inds:
                if filo[:,0][ind] not in ticid:
                    ticid.append(int(filo[:,0][ind]))
                    otype.append(filo[:,1][ind])
                    asasid.append(filo[:,2][ind])
        print('ASAS-SN cross matched targets: '+str(len(ticid)))

        # -- get periods, amplitudes from ASAS-SN ------------------------------
        df = pd.read_csv(metapath+'asassn_catalog.csv')
        amplitude = []
        period = []
        for i in range(len(ticid)):
            ind = df['asassn_name']==asasid[i]
            amplitude.append(list(df['amplitude'][ind])[0])
            period.append(list(df['period'][ind])[0])

        np.savetxt(savepath+'lspm-feats-mock/asas_sn_period_amplitude.txt',
                   np.array([ticid, otype, asasid, amplitude, period]).T,
                   fmt='%s', delimiter=',')

    # -- randomly sample ASAS-SN targets ---------------------------------------
    with open(fname_asas, 'w') as f:
        line = 'Hyperparam_Set,TICID,thresh_max,thresh_min,kernel,'+\
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

        for i in range(batch_size): 
            # >> randomly pick light curve
            ind = np.random.choice(np.arange(len(ticid)))
            t, y = fe.get_lc(datapath+'mask/', ticid[ind], timescale, rmv_nan=True)
            lspm = fits.open(datapath+'timescale-'+str(timescale)+'sector/lspm/'+\
                             str(ticid[ind])+'.fits')
            frequency = lspm[1].data['FREQ']
            power = lspm[1].data['LSPM']
            
            P_pred, A_pred, lspm_feats = fe.peak_finder(t=t, y=y,
                                                  frequency=frequency,
                                                  power=power, datapath=datapath,
                                                  ticid=ticid[ind],
                                                  timescale=timescale,
                                                  savepath=savepath)
            # >> Hyperparam_Set,Light_Curve,
            line = ','.join(map(str, [hpset, ticid[ind], h1, h2, h3, h4]))+','
            if len(P_pred) > 0:
                line += ','.join(map(str, [period[ind], P_pred[0], amplitude[ind],
                                           A_pred[0]]))
            else:
                line += ','.join(map(str, [period[ind], np.nan, amplitude[ind],
                                           np.nan]))
            with open(fname_asas, 'a') as f:
                f.write(line+'\n') 


if plot:
    with open(fname_mock, 'r') as f:
        cols = f.readline()[:-1].split(',') # >> column names
    data = np.loadtxt(fname_mock, skiprows=1, delimiter=',')
    err = [] # >> mean square error

    # -- Create correlation matrix ---------------------------------------------
    feat_ind = 9
    mat_cols = cols[feat_ind-4:feat_ind]
    for i in range(int((len(cols)-feat_ind)/2)): # >> loop through features
                                          # >> (P, Pratio, etc.)
        ind = feat_ind+(i*2)
        err.append(np.abs(data[:,ind+1]-data[:,ind]))
        mat_cols.append('_'.join(cols[ind].split('_')[:-1])+'_err')
    mat_data = np.concatenate([data[:,feat_ind-4:feat_ind], np.array(err).T], axis=1)
    pt.plot_corr_matrix(mat_data, savepath+'lspm-feats-mock/', cols=mat_cols,
                        annot_kws={'size': 'xx-small'})

    # -- best hyperparameter set -----------------------------------------------
    # err = data[:,np.nonzero(np.array(mat_cols)=='P_skew_err')[0][0]]
    err = data[:,np.nonzero(np.array(mat_cols)=='A_skew_err')[0][0]]
    err = err[:(len(err)//batch_size)*batch_size]
    err = np.mean(np.split(err, len(err)//batch_size), axis=-1)
    best_ind = np.argmin(err)
    best_hpset = hpsets[best_ind]
    print('Best hyperparameter set :'+str(best_hpset))

    # -- precision, recall plots -----------------------------------------------
    frequency = fe.get_frequency_grid('/scratch/echickle/', timescale)
    df = frequency[1] - frequency[0]
    n_samples = 1000
    h1, h2, h3, h4 = best_hpset
    t_batch, y_batch, period, amp, mag = \
        fe.periodic_simulated_data(savepath+'lspm-feats-mock/', datapath,
                                   metapath, kernel=h3, timescale=1,
                                   freq=frequency,
                                   n_samples=n_samples)    
    precision, recall = [], []
    for i in range(n_samples):
        power = LombScargle(t_batch[i], y_batch[i], nterms=h4).power(frequency)   
        P_pred, A_pred, lspm_feats = \
        fe.peak_finder(t=t_batch[i], y=y_batch[i], frequency=frequency,
                       power=power, datapath=datapath, fmax=1/(1/24.),
                       thresh_max=h1, thresh_min=h2, n_terms=h4,
                       timescale=timescale, plot=False)
        p, r = fe.eval_simulated_data(P_pred, period[i], df)
        precision.append(p)
        recall.append(r)

    # >> bin precision and recall based on magnitude
    precision, recall = np.array(precision), np.array(recall)
    binned_mag = np.unique(mag)
    binned_mag.sort()
    
    binned_precision = []
    binned_recall = []
    for m in binned_mag:
        inds = np.nonzero(binned_mag==m)
        binned_precision.append(np.mean(precision[inds]))
        binned_recall.append(np.mean(recall[inds]))
    binned_mag = np.mean(np.split(binned_mag, 100), axis=1)
    binned_precision = np.mean(np.split(np.array(binned_precision), 100), 
                               axis=1)
    binned_recall = np.mean(np.split(np.array(binned_recall), 100), axis=1)

    inds = np.nonzero(data[:,0]==best_ind)
    plt.figure()
    # plt.plot(mag, precision, '.')
    plt.plot(binned_mag, binned_precision, '.', ms=4)
    plt.xlabel('Magnitude')
    plt.ylabel('Precision of periods')
    plt.tight_layout()
    plt.savefig(savepath+'lspm-feats-mock/mag_precision.png')
    print(savepath+'lspm-feats-mock/mag_precision_'+min_err+'.png')

    plt.figure()
    # plt.plot(mag, recall, '.')
    plt.plot(binned_mag, binned_recall, '.', ms=4)
    plt.xlabel('Magnitude')
    plt.ylabel('Recall of periods')
    plt.tight_layout()
    plt.savefig(savepath+'lspm-feats-mock/mag_recall.png')
    print(savepath+'lspm-feats-mock/mag_recall_'+min_err+'.png')

if plot_asassn:    

    data = np.loadtxt(savepath+'lspm-feats-mock/asas_sn_period_amplitude.txt',
                      dtype='str', delimiter=',')
    ticid = data[:,0].astype('int') 
    otype, asasid = data[:,1], data[:,2]
    amplitude, period = data[:,3].astype('float'), data[:,4].astype('float')

    # >> remove targets that have periods longer than TESS baseline
    inds = np.nonzero(period <= 27.)
    ticid, otype, asasid = ticid[inds], otype[inds], asasid[inds]
    amplitude, period = amplitude[inds], period[inds]

    with open(fname_asas, 'r') as f:
        cols = f.readline()[:-1].split(',') # >> column names
    data = np.loadtxt(fname_mock, skiprows=1, delimiter=',')
    err = [] # >> mean square error

    # -- correlation matrix ----------------------------------------------------
    feat_ind = 6
    mat_cols = cols[feat_ind-4:feat_ind]
    for i in range(int((len(cols)-feat_ind)/2)): 
        ind = feat_ind+(i*2)
        err.append(np.abs(data[:,ind+1]-data[:,ind]))
        mat_cols.append('_'.join(cols[ind].split('_')[:-1])+'_err')
    mat_data = np.concatenate([data[:,feat_ind-4:feat_ind], np.array(err).T], axis=1)
    pt.plot_corr_matrix(mat_data, savepath+'lspm-feats-mock/asas_sn_',
                        cols=mat_cols, annot_kws={'size': 'xx-small'})

    # -- best hyperparameter set -----------------------------------------------
    # err = data[:,np.nonzero(np.array(mat_cols)=='P_skew_err')[0][0]]
    err = data[:,np.nonzero(np.array(mat_cols)=='P_err')[0][0]]
    err = err[:(len(err)//batch_size)*batch_size]
    err = np.mean(np.split(err, len(err)//batch_size), axis=-1)
    best_ind = np.argmin(err)
    best_hpset = hpsets[best_ind]
    print('Best hyperparameter set :'+str(best_hpset))

    # -- bar plot of error vs. otype -------------------------------------------
    h1, h2, h3, h4 = best_hpset

    unq_otypes, counts = np.unique(otype, return_counts=True)
    inds = np.argsort(counts)[::-1]
    unq_otypes, counts = unq_otypes[inds], counts[inds]
    max_bars = len(unq_otypes)
    avg_err = []
    for i in range(max_bars):
        err = []
        inds = np.nonzero(otype == unq_otypes[i])[0][:batch_size]
        for ind in inds:
            t, y = fe.get_lc(datapath+'mask/', ticid[ind], timescale,
                             rmv_nan=True)            
            lspm = fits.open(datapath+'timescale-'+str(timescale)+'sector/lspm/'+\
                             str(ticid[ind])+'.fits')
            frequency = lspm[1].data['FREQ']
            power = lspm[1].data['LSPM']
            
            P_pred, A_pred, lspm_feats = fe.peak_finder(t=t, y=y,
                                                  frequency=frequency,
                                                  power=power, datapath=datapath,
                                                  ticid=ticid[ind],
                                                  timescale=timescale,
                                                  savepath=savepath)  
            if len(P_pred) > 0:
                err.append(np.abs(P_pred[0]-period[ind]))
            else:
                err.append(np.nan)
        avg_err.append(np.nanmean(err))
        
    fig,ax = plt.subplots(figsize=(15,8))
    inds = np.nonzero(~np.isnan(avg_err))[0]
    ax.bar(unq_otypes[:max_bars][inds],np.array(avg_err)[inds])
    ax.set_xlabel('Object types')
    ax.set_ylabel('Average absolute error of periods')
    fig.tight_layout()
    fig.savefig(savepath+'lspm-feats-mock/asas_sn_bar.png')
    print('Saved '+savepath+'lspm-feats-mock/asas_sn_bar.png')
