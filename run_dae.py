"""
Created on 210905

run_dae.py
Script to train a deep autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import feat_eng as fe

# -- inputs --------------------------------------------------------------------

datapath = '/scratch/data/tess/lcur/spoc/'
savepath = '/scratch/echickle/'
parampath    = '/home/echickle/work/daehyperparams.txt'
datatype     = 'SPOC'
featgen      = 'DAE'
sectors      = list(range(1,27))

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype, sectors=sectors,
            parampath=parampath)

# >> Load LS periodograms
fe.load_lspgram(mg)

# >> Feature extraction by deep autoencoder
mg.generate_dae_features()
# mg.load_features("DAE")

mg.run_feature_analysis("DAE")

# pdb.set_trace()
# mg.load_lightcurves_local(mg.datapath+'mask/', 1)
mg.run_vis("DAE")

# mg.load_reconstructions("DAE")
# mg.produce_ae_visualizations("DAE")


# mg.load_lightcurves_local(mg.datapath+'clip/')
# mg.preprocess_data(featgen)

# mg.load_features(featgen)
# mg.run_feature_analysis(featgen)

# mg.load(featgen)

# mg.produce_novelty_visualizations(featgen)

# mg.run_vis(featgen)
# mg.produce_ae_visualizations(featgen)

