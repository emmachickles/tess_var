"""
Created on 210728

run_vis.py
Script to produce pipeline visualizations
@author: Emma Chickles (emmachickles)
"""

from __init__ import *

# -- inputs --------------------------------------------------------------------
datapath     = '/nfs/blender/data/tdaylan/data/'
savepath     = '/nfs/blender/data/tdaylan/Mergen_Run_1/'
datatype     = 'SPOC'
sector       = 1
runiter      = True
numiter      = 1
ncomp        = 3
perplxty     = 70
anim         = True
elev         = 45

# -- initialize Mergen object --------------------------------------------------
mg = mergen.mergen(datapath, savepath, datatype, sector=sector,
                   runiter=runiter, numiter=numiter)


mg.load_lightcurves_local()
mg.load_learned_features()
mg.load_true_otypes()
mg.numerize_true_otypes()

# mg.generate_clusters()
mg.load_gmm_clusters()

# mg.generate_predicted_otypes()
mg.load_pred_otypes()

mg.numerize_pred_otypes()

# mg.generate_tsne()
# mg.produce_clustering_visualizations()

mg.generate_novelty_scores()
mg.produce_novelty_visualizations()
