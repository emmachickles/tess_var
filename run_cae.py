"""
Created on 211105

run_cae.py
Script to train a convolutional autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import feat_eng as fe

# -- inputs --------------------------------------------------------------------

datapath     = '/scratch/data/tess/lcur/spoc/'
savepath     = '/scratch/echickle/'
metapath     = '/scratch/data/tess/meta/'
parampath    = '/home/echickle/work/caehyperparams.txt'
datatype     = 'SPOC'
featgen      = 'CAE'
clstrmeth    = 'gmm'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
numclstr     = 300

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype, metapath=metapath,
            parampath=parampath, featgen=featgen, clstrmeth=clstrmeth,
            mdumpcsv=mdumpcsv, numclstr=numclstr)

fe.load_lspgram_fnames(mg)
mg.generate_features() # >> feature extraction by conv autoencoder
mg.generate_clusters()
