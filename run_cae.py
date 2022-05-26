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

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype,  parampath=parampath,
            featgen=featgen, clstrmeth=clstrmeth,
            numclstr=numclstr, name=train_on_, metapath=metapath,
            mdumpcsv=mdumpcsv)

fe.load_lspgram_fnames(mg, timescale=timescale)
# mg.optimize_params()
# mg.generate_features(save=False) # >> feature extraction by conv autoencoder
# mg.save_ae_features(reconstruct=False)

mg.load_features()
mg.generate_tsne()
mg.generate_clusters()
