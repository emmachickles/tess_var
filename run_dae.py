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
metapath = '/scratch/data/tess/meta/'
parampath    = '/home/echickle/work/daehyperparams.txt'
datatype     = 'SPOC'
featgen      = 'DAE'
# clstrmeth    = 'hdbscan'
clstrmeth    = 'gmm'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
numclstr     = 300

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype, metapath=metapath,
            parampath=parampath, featgen=featgen, clstrmeth=clstrmeth,
            mdumpcsv=mdumpcsv, numclstr=numclstr)

fe.load_lspgram_fnames(mg, lspmpath='lspm-1sector/')

# !! >>>
params = lt.read_hyperparameters_from_txt(mg.parampath)
model = lt.load_model(mg.savepath+'model/model.hdf5')
lt.save_autoencoder_products(model, params, mg.batch_fnames, mg.savepath+'model/')
# !! <<<

mg.generate_features() # >> feature extraction by deep autoencoder
# mg.load_features()

mg.generate_clusters()
# mg.load_gmm_clusters()
mg.load_nvlty()

mg.load_true_otypes()

mg.generate_predicted_otypes()
mg.numerize_otypes()
mg.load_tsne()

mg.run_vis() # !! add classification plots, etc

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

