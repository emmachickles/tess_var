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
output_dir   = './'
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

# mg.generate_clusters()
mg.load_gmm_clusters()

mg.load_vtypes()
mg.numerize_vtypes()

# -- produce t-SNE -------------------------------------------------------------
# X = lt.run_tsne(mg.features, n_components=ncomp, perplexity=perplxty,
#                 savepath=mg.ensbpath)
X = lt.load_tsne(mg.ensbpath)

# -- produce plots -------------------------------------------------------------
# prefix = 'perplexity'+str(perplexity)+'_elev'+str(elev)+'_'

# >> color with clustering results
# prefix = 'pred_'
# pt.plot_tsne(mg.feats, mg.clstr, X=X, output_dir=output_dir,
#              prefix=prefix, animate=anim, elev=elev)

# >> color with classifications from GCVS, SIMBAD, ASAS-SN
prefix = 'true_'
pt.plot_tsne(mg.feats, mg.numvtype, X=X, output_dir=output_dir,
             prefix=prefix, animate=anim, elev=elev, uniqvtype=mg.uniqvtype)

