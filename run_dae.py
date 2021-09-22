"""
Created on 210905

run_dae.py
Script to train a deep autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *

# -- inputs --------------------------------------------------------------------
# datapath     = '/Users/emma/Documents/MIT/TESS/Data/'
datapath     = '/nfs/blender/data/tdaylan/data/'

# savepath     = '/Users/emma/Documents/MIT/TESS/Plots/Ensemble-Sector_1/DAE/'
savepath     = '/nfs/blender/data/tdaylan/Mergen_Run_2/'

parampath    = datapath + 'daehyperparams.txt'
datatype     = 'SPOC'
featgen      = 'DAE'
sector       = 1

# -- initialize Mergen object --------------------------------------------------
mg = mergen.mergen(datapath, savepath, datatype, sector=sector,
                   parampath=parampath)

mg.load_lightcurves_local()
mg.preprocess_data(featgen)

# mg.load_features(featgen)
# mg.run_feature_analysis(featgen)

mg.load(featgen)

mg.produce_novelty_visualizations(featgen)

# mg.run_vis(featgen)
# mg.produce_ae_visualizations(featgen)

