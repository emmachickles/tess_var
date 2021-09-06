"""
Created on 210905

run_dae.py
Script to train a deep autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *

# -- inputs --------------------------------------------------------------------
datapath     = '/nfs/blender/data/tdaylan/data/'
savepath     = '/nfs/blender/data/tdaylan/Mergen_Run_2/'
parampath    = './daehyperparams.txt'
datatype     = 'SPOC'
sector       = 1

# -- initialize Mergen object --------------------------------------------------
mg = mergen.mergen(datapath, savepath, datatype, sector=sector,
                   parampath=parampath)

mg.load_lightcurves_local()
mg.preprocess_dae()
mg.load_features('DAE')
# mg.generate_dae_features()

mg.load_true_otypes()
mg.numerize_true_otypes()

mg.generate_clusters()
mg.generate_predicted_otypes()
mg.numerize_pred_otypes()

mg.generate_tsne()
mg.produce_clustering_visualizations()

mg.generate_novelty_scores()
mg.produce_novelty_visualizations()
