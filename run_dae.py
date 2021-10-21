"""
Created on 210905

run_dae.py
Script to train a deep autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import feat_eng as fe

# -- inputs --------------------------------------------------------------------
# datapath     = '/Users/emma/Documents/MIT/TESS/Data/'
# datapath     = '/nfs/blender/data/tdaylan/data/'
datapath = '/scratch/data/tess/lcur/spoc/'

# savepath     = '/Users/emma/Documents/MIT/TESS/Plots/Ensemble-Sector_1/DAE/'
# savepath     = '/nfs/blender/data/tdaylan/Mergen_Run_2/'
savepath = '/scratch/echickle/'

parampath    = '/home/echickle/work/daehyperparams.txt'
datatype     = 'SPOC'
featgen      = 'DAE'
sector       = 1

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype,
                   parampath=parampath)

# fe.preprocess_lspgram(mg)
fe.load_lspgram(mg)
mg.generate_dae_features()
mg.run_feature_analysis("DAE")
mg.run_vis("DAE")

# mg.load_lightcurves_local(mg.datapath+'clip/')
# mg.preprocess_data(featgen)

# mg.load_features(featgen)
# mg.run_feature_analysis(featgen)

# mg.load(featgen)

# mg.produce_novelty_visualizations(featgen)

# mg.run_vis(featgen)
# mg.produce_ae_visualizations(featgen)

