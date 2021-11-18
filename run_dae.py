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
sector       = list(range(1,27))

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype, sector=sector, metapath=metapath,
            parampath=parampath, featgen=featgen)

fe.load_lspgram_fnames(mg)
# params = lt.read_hyperparameters_from_txt(mg.parampath)

# # !! >>>
# model = lt.load_model(mg.featpath+'model.hdf5')
# params = lt.read_hyperparameters_from_txt(mg.parampath)
# lt.save_autoencoder_products(model, params, mg.batch_fnames, mg.featpath)
# # !! <<<


# fe.load_lspgram(mg) # >> load ls periodograms

# mg.generate_features() # >> feature extraction by deep autoencoder
mg.load_features()

# mg.feats = np.load(mg.featpath+'chunk00_bottleneck_train.npy')
# mg.sector = np.load(mg.datapath+'dae/chunk00_train_sector.npy')
# mg.objid = np.load(mg.datapath+'dae/chunk00_train_ticid.npy')


mg.run_feature_analysis()

# pdb.set_trace()
# mg.load_lightcurves_local(mg.datapath+'mask/', 1)
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

