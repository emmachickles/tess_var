"""
Created on 210905

run_dae.py
Script to train a deep autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import feat_eng as fe

# -- inputs --------------------------------------------------------------------

# >> timescale : int, number of sectors (1, 6, 13)
timescale    = 1
savepath     = '/scratch/echickle/timescale-'+str(timescale)+'sector/'
dt.create_dir(savepath)

train_on_ = 'stat' # 'stat', 'lspm'

datapath = '/scratch/data/tess/lcur/spoc/'
metapath = '/scratch/data/tess/meta/'

datatype     = 'SPOC'
featgen      = 'DAE'
# clstrmeth    = 'hdbscan'
clstrmeth    = 'gmm'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
numclstr     = 300

if train_on_ == 'lspm':
    parampath = '/home/echickle/work/daehyperparams.txt'
if train_on_ == 'stat':
    parampath = '/home/echickle/work/daestathyperparams.txt'

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype, metapath=metapath,
            parampath=parampath, featgen=featgen, clstrmeth=clstrmeth,
            mdumpcsv=mdumpcsv, numclstr=numclstr, name=train_on_)

if train_on_ == 'lspm':
    fe.load_lspgram_fnames(mg, timescale=timescale) 
if train_on_ == 'stat':
    fe.load_stats(mg, timescale=timescale)

mg.generate_features() # >> feature extraction by deep autoencoder
# lt.save_autoencoder_products(parampath=mg.parampath,
#                              output_dir=mg.featpath+'model/',
#                              batch_fnames=mg.batch_fnames)
# pdb.set_trace()
# mg.load_features()

mg.generate_tsne()
# mg.load_tsne()


mg.generate_clusters()
pdb.set_trace()

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

