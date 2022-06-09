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
    savepath     = '/work/submit/echickle/timescale-'+str(timescale)+'sector/'
    dt.create_dir(savepath)

    datapath     = '/work/submit/echickle/data/'
    metapath     = None
    mdumpcsv     = None

    parampath    = '/home/submit/echickle/work/tess_var/docs/hyperparam.txt'

datatype     = 'SPOC'
featgen      = 'CAE'
clstrmeth    = 'gmm'
numclstr     = 300
train_on_    = 'lspm'
# train_on_ = 'unreg-lspm'

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype,  parampath=parampath,
            featgen=featgen, clstrmeth=clstrmeth,
            numclstr=numclstr, name=train_on_, metapath=metapath,
            mdumpcsv=mdumpcsv)

fe.load_lspgram_fnames(mg, timescale=timescale)
# mg.optimize_params()
# mg.generate_features(save=False) # >> feature extraction by conv autoencoder
# mg.save_ae_features(reconstruct=False)

# lt.save_autoencoder_products(batch_fnames=mg.batch_fnames, parampath=parampath,
#                              output_dir=mg.featpath+'model/')

mg.load_features()
# mg.generate_tsne()
# mg.generate_clusters()
# mg.load_tsne()
mg.load_gmm_clusters()

mg.load_true_otypes() # >> return numtot, otdict
mg.generate_predicted_otypes()
rec, fdr, pre, acc, cnts_true, cnts_pred = mg.evaluate_classification()

# mg.numerize_otypes()
# mg.produce_clustering_visualizations()
# mg.run_vis()

# lt.label_clusters_2(mg)
