"""
Created on 211105

run_cae.py
Script to train a convolutional autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

print('run_cae.py')

from __init__ import *
import feat_eng as fe
import tensorflow as tf
import matplotlib.pyplot as plt

# -- inputs --------------------------------------------------------------------

timescale    = 1 # >> number of sectors (1, 6, or 13)

machine      = 'submit'

if machine == 'uzay':
    savepath     = '/scratch/echickle/timescale-'+str(timescale)+'sector/'
    dt.create_dir(savepath)

    datapath     = '/scratch/data/tess/lcur/spoc/'
    metapath     = '/scratch/data/tess/meta/'
    mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'

    parampath    = '/home/echickle/work/tess_var/docs/hyperparam.txt'
    

elif machine == 'submit':
    savepath = '/data/submit/echickle/phase_curve/'
    savepath     = '/work/submit/echickle/timescale-'+str(timescale)+'sector/'
    dt.create_dir(savepath)

    datapath     = '/work/submit/echickle/data/'
    metapath     = None
    mdumpcsv     = None

    parampath    = '/home/submit/echickle/work/tess_var/docs/hyperparam.txt'

datatype     = 'SPOC'
featgen      = 'CAE'
clstrmeth    = 'gmm'
numclstr     = 250
train_on_    = 'lspm'
# train_on_ = 'unreg-lspm'

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype,  parampath=parampath,
            featgen=featgen, clstrmeth=clstrmeth,
            numclstr=numclstr, name=train_on_, metapath=metapath,
            mdumpcsv=mdumpcsv)

# fe.load_lspgram_fnames(mg, timescale=timescale)

ae_path = '/data/submit/echickle/ae/'
mg.batch_fnames = []
mg.sector = []
mg.objid = []
mg.x_train = None
for i in range(10):
    mg.batch_fnames.extend(ae_path+'chunk%02d'%i+'_train_lspm.npy')
    mg.objid.extend(np.load(ae_path + 'chunk%02d'%i+'_train_ticid.npy'))
    mg.sector.extend(np.load(ae_path + 'chunk%02d'%i+'_train_sector.npy') )
mg.sector = np.array(mg.sector)
mg.objid = np.array(mg.objid)
print(mg.batch_fnames)
# mg.optimize_params()
mg.generate_features() # >> feature extraction by conv autoencoder
# mg.save_ae_features(reconstruct=False)
# lt.save_autoencoder_products(batch_fnames=mg.batch_fnames, parampath=parampath,
#                              output_dir=mg.featpath+'model/')

# mg.load_features()
# mg.generate_tsne()

# mg.load_tsne()
# mg.numclstr = None
# mg.generate_clusters()
# mg.load_gmm_clusters()

# targ = np.loadtxt('/scratch/data/tess/meta/spoc/obj/ROT_Kochukhov_2021.txt')
# centr, dist = lt.clstr_centr(mg.feats)

# inter, comm1, comm2 = np.intersect1d(mg.objid, targ, return_indices=True)
# targ_clstr = mg.clstr[comm1]

# c = 239
# targ_c = targ[np.nonzero(targ_clstr == c)]

# inds_clstr = np.nonzero(mg.clstr == c)

# mg.load_true_otypes() 


# for numclstr in [10, 25, 50, 100, 150, 200, 250, 300]:
#     mg.numclstr=numclstr
#     mg.load_gmm_clusters()

#     recall = []
#     for i in np.unique(mg.numtot):
#         unq, cnt = np.unique(mg.clstr[np.nonzero(mg.numtot == i)], return_counts=True)
#         recall.append(max(cnt)/np.sum(cnt))
#     recall = np.array(recall)
#     txt = np.array([np.unique(mg.numtot), np.array(list(mg.otdict.values())),
#                     recall]).T
#     np.savetxt(mg.featpath+'gmm_'+str(mg.numclstr)+'_recall.txt', txt, fmt='%s')

#     # print(

# mg.generate_predicted_otypes()



# rec, fdr, pre, acc, cnts_true, cnts_pred = mg.evaluate_classification()

# mg.numerize_otypes()
# mg.produce_clustering_visualizations()
# mg.run_vis()
# pt.clstr_centr_hist(mg.feats, mg.clstr, mg.totype, mg.featpath+'hist/', facecolor='k', textcolor='w', histcolor='#cfe2f3ff')

# lt.label_clusters_2(mg)

# import pandas as pd

# fe.load_lspgram(mg)

# tic_cat = pd.read_csv(mg.metapath+'spoc/tic/sector-01-tic_cat.csv', index_col=False)
# recn = []
# for i in range(10):
#     recn.extend(np.load(mg.featpath+'model/chunk%02d'%i+'_x_predict_train.npy'))
# recn = np.array(recn)

# inter, comm1, comm2 = np.intersect1d(tic_cat['ID'], mg.objid, return_indices=True)

# tic_cat = tic_cat.iloc[comm1]

# mg.objid = mg.objid[comm2]
# mg.x_train = mg.x_train[comm2]
# mg.x_train = mg.x_train[:,-recn.shape[1]:]
# mg.freq = mg.freq[-recn.shape[1]:]
# recn = recn[comm2]
# err = np.mean( (mg.x_train - recn)**2, axis=1 )


# cols = ['Tmag', 'Teff', 'rad', 'mass', 'logg', 'rho']
# for i in range(len(cols)):
#     plt.figure()
#     plt.title(cols[i])
#     plt.plot(err, tic_cat[cols[i]], '.', ms=1)
#     plt.xscale('log')
#     plt.xlabel('Reconstruction error (MSE)')
#     plt.ylabel(cols[i])
#     plt.savefig(mg.featpath+'diag/err_'+cols[i]+'.png')
#     plt.close()

# for i in range(10):
#     ind = np.argsort(err)[i]
#     plt.figure()
#     plt.title('MSE '+str(round(err[ind], 3)))
#     plt.plot(mg.freq, mg.x_train[ind], 'k', label='Input') 
#     plt.plot(mg.freq, recn[ind], label='Reconstruction') 
#     plt.legend()
#     plt.xscale('log')
#     plt.savefig(mg.featpath+'diag/err_low_'+str(i)+'.png')
#     plt.close()

#     ind = np.argsort(err)[-i-1]
#     plt.figure()
#     plt.title('MSE '+str(round(err[ind], 3)))
#     plt.plot(mg.freq, mg.x_train[ind], 'k', label='Input') 
#     plt.plot(mg.freq, recn[ind], label='Reconstruction') 
#     plt.legend()
#     plt.xscale('log')
#     plt.savefig(mg.featpath+'diag/err_high_'+str(i)+'.png')
#     plt.close()

    

