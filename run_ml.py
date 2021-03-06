"""
Created on 211105

run_ml.py
Script to train a convolutional autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import feat_eng as fe
import tensorflow as tf

# -- inputs --------------------------------------------------------------------

timescale    = 1 # >> number of sectors (1, 6, or 13)
savepath     = '/work/submit/echickle/timescale-'+str(timescale)+'sector/'
dt.create_dir(savepath)

datapath     = '/work/submit/echickle/data/'
metapath     = '/scratch/data/tess/meta/'
parampath    = '/home/submit/echickle/work/tess_var/docs/hyperparam.txt'
datatype     = 'SPOC'
# featgen      = 'CAE'
featgen      = 'DAE'
# clstrmeth    = 'hdbscan'
clstrmeth    = 'gmm'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
numclstr     = 100
# name         = 'cvae-lspm'
name         = 'lspm'

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype, metapath=metapath,
            featgen=featgen, clstrmeth=clstrmeth, parampath=parampath,
            mdumpcsv=mdumpcsv, numclstr=numclstr, name=name)

fe.load_lspgram_fnames(mg, timescale=timescale)
mg.generate_features() # >> feature extraction by conv autoencoder
# mg.load_features()
mg.generate_clusters()

