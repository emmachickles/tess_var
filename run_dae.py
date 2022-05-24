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

train_on_ = 'lspm' # 'stat', 'lspm'

datapath = '/scratch/data/tess/lcur/spoc/'
metapath = '/scratch/data/tess/meta/'

datatype     = 'SPOC'
featgen      = 'DAE'
clstr    = 'gmm'
# clstrmeth    = 'gmm'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
numclstr     = 300

if train_on_ == 'lspm':
    parampath = '/home/echickle/work/daehyperparams.txt'
if train_on_ == 'stat':
    parampath = '/home/echickle/work/daestathyperparams.txt'

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype, metapath=metapath,
            featgen=featgen, clstrmeth='gmm',
            mdumpcsv=mdumpcsv, numclstr=numclstr, name=train_on_)
# parampath=parampath

if train_on_ == 'lspm':
    fe.load_lspgram_fnames(mg, timescale=timescale) 
if train_on_ == 'stat':
    fe.load_stats(mg, timescale=timescale)

# !! tmp
mg.featpath = '/scratch/echickle/timescale-1sector/DAE/'
txt = np.loadtxt(mg.featpath+'gmm_labels.txt')
# ticid_cluster = txt[0].astype('int')
# sector_cluster = []
# for i in range(len(ticid_cluster)):
#     if ticid_cluster[i] in mg.objid:
#         sector_cluster.append(mg.sector[np.nonzero(mg.objid == ticid_cluster[i])])
mg.objid = txt[0].astype('int')

# mg.generate_features() # >> feature extraction by deep autoencoder
# lt.save_autoencoder_products(parampath=mg.parampath,
#                              output_dir=mg.featpath+'model/',
#                              batch_fnames=mg.batch_fnames)
# pdb.set_trace()
# mg.load_features()

# mg.generate_tsne()
# mg.load_tsne()

mg.load_features()
# mg.generate_clusters()
# pdb.set_trace()

# mg.load_gmm_clusters()
# mg.load_nvlty()

# from sklearn.manifold import TSNE
# X = TSNE(n_components=3).fit_transform(mg.feats)
# np.save(mg.featpath+'tsne_dim3.npy', X)
X = np.load(mg.featpath+'tsne_dim3.npy')

mg.load_true_otypes()
print('plot')
prefix = 'true_'
pt.plot_tsne(mg.feats, mg.numtot, X=X, output_dir=mg.featpath, prefix=prefix, animate=True, otypedict=mg.otdict)

pdb.set_trace()
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

