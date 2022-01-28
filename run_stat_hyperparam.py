"""
Created on 220115

run_stat_hyperparam.py
Script to produce simulated data and perform a hyperparameter search for the
LS-periodogram statistics.

@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import feat_eng as fe

# >> what to run?
prep_spoc    = True  # >> preprocesses all short-cadence SPOC data
target_prep  = False # >> preprocess a single target
diag_only    = False # >> produce diagnostic plots

# >> inputs
datapath     = '/scratch/data/tess/lcur/spoc/'
savepath     = '/scratch/echickle/'
metapath     = '/scratch/data/tess/meta/'
timescale    = 1
datatype     = 'SPOC'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
featgen      = 'DAE'

# # ------------------------------------------------------------------------------

# # >> Initialize Mergen object
mg = mergen(datapath, savepath, datatype, metapath=metapath,
            mdumpcsv=mdumpcsv, featgen=featgen, timescale=timescale)

# fe.spoc_noise(datapath, metapath, savepath)
fe.periodic_simulated_data(savepath, datapath, metapath)

# fe.periodic_simulated_data(savepath, datapath, metapath, timescale=1)

# # >> Let's plot some light curves from Sector 1 based on Tmag
# lcpath = datapath+'clip/'
# sector = 1
# filo = np.loadtxt(metapath+'spoc/targ/2m/all_targets_S%03d'%sector+'_v1.txt')
# ticid = filo[:,0].astype('int')
# Tmag = filo[:,3]

# ticid_lc = [int(f[:-5]) for f in os.listdir(lcpath+'sector-%02d'%sector+'/')]
# _, comm1, comm2 = np.intersect1d(ticid, ticid_lc, return_indices=True)
# ticid = ticid[comm1]
# Tmag = Tmag[comm1]

# inds = np.argsort(Tmag)
# ticid = ticid[inds]
# Tmag = Tmag[inds]
# grid = np.indices(ticid.shape)[0]

# nrows = 10
# fig, ax = plt.subplots(nrows, 3, figsize=(nrows*4, 3*10))
# for i, inds in zip(range(3), [grid[0:nrows], grid[8000:8000+nrows], grid[-nrows:]]):
#     for j in range(len(inds)):
#         t, y = fe.get_lc(lcpath, ticid[inds[j]], 1, return_sector=True, rmv_nan=True)
#         t, y = t[0], y[0]
#         fe.plot_lc(ax[j,i], t, y)
#         std = fe.local_std(t, y, 10)
#         ax[j,i].set_title('TIC '+str(ticid[inds[j]])+\
#                           '\nTmag '+str(Tmag[inds[j]])+\
#                           ', Local STD '+str(np.round(std, 3)))
# fig.tight_layout()
# fig.savefig(savepath+'timescale-1sector/feat_eng/lspm-feats-mock/Tmag.png')





