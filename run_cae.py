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
parampath    = '/home/echickle/work/caehyperparams.txt'
datatype     = 'SPOC'
featgen      = 'CAE'
sector       = list(range(1,27))

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype, sector=sector,
            parampath=parampath, featgen=featgen)

fe.load_lspgram(mg) # >> load ls periodograms
# fe.load_lspgram_fnames(mg)
mg.generate_features() # >> feature extraction by conv autoencoder
