"""
Created on 211105

run_cae.py
Script to train a convolutional autoencoder on LS periodograms.
@author: Emma Chickles (emmachickles)
"""

print('run_cae.py')

from __init__ import *
import feat_eng as fe
import tensorflow as tf

# -- inputs --------------------------------------------------------------------

timescale    = 1 # >> number of sectors (1, 6, or 13)

machine      = 'uzay'

if machine == 'uzay':
    savepath     = '/scratch/echickle/timescale-'+str(timescale)+'sector/'
    dt.create_dir(savepath)

    datapath     = '/scratch/data/tess/lcur/spoc/'
    metapath     = '/scratch/data/tess/meta/'
    mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'

    parampath    = '/home/echickle/work/tess_var/docs/hyperparam.txt'
    

elif machine == 'submit':
    savepath     = '/work/submit/echickle/timescale-'+str(timescale)+'sector/'
    dt.create_dir(savepath)

    datapath     = '/work/submit/echickle/data/'
    metapath     = None
    mdumpcsv     = None

    parampath    = '/home/submit/echickle/work/tess_var/docs/hyperparam.txt'

datatype     = 'SPOC'
featgen      = 'CAE'
clstrmeth    = 'gmm'
numclstr     = 300
train_on_    = 'lspm'
# train_on_ = 'unreg-lspm'

# -- initialize Mergen object --------------------------------------------------
mg = mergen(datapath, savepath, datatype,  parampath=parampath,
            featgen=featgen, clstrmeth=clstrmeth,
            numclstr=numclstr, name=train_on_, metapath=metapath,
            mdumpcsv=mdumpcsv)

fe.load_lspgram_fnames(mg, timescale=timescale)
# mg.optimize_params()
# mg.generate_features(save=False) # >> feature extraction by conv autoencoder
# mg.save_ae_features(reconstruct=False)

mg.load_features()
# mg.generate_tsne()
# mg.generate_clusters()

mg.load_tsne()
mg.load_gmm_clusters()
# mg.load_true_otypes()
# 
# mg.numerize_otypes()
# mg.produce_clustering_visualizations()
# prefix = 'gmm300_'
# pt.plot_tsne(mg.feats, mg.clstr, X=mg.tsne, output_dir=mg.featpath,
#           prefix=prefix)

# unique, counts = np.unique(mg.numtot, return_counts=True)
# inds = np.nonzero(mg.numtot == np.argmax(counts))
# mg.numtot[inds] = 3000

# prefix = 'true_'
# pt.plot_tsne(mg.feats, mg.numtot, X=mg.tsne, output_dir=mg.featpath,
#           prefix=prefix)

# prefix = 'inset_'
# pt.plot_tsne(mg.feats, mg.clstr, X=mg.tsne, output_dir=mg.featpath+'imgs/',
#              prefix=prefix, plot_insets=True, objid=mg.objid, 
#              lcpath=mg.datapath+'clip/', 
#              max_insets=300)


# prefix='crot_'
# # >> known complex rotators (sectors 1, 2)
# # targets      = [38820496, 177309964, 206544316, 234295610, 289840928,
# #                 425933644, 425937691] # >> sector 1 only
# targets = [38820496, 177309964, 201789285, 206544316, 224283342,\
#            234295610, 289840928, 332517282, 425933644, 425937691]
# numcot = [] # >> numerized complex rotator ensemble object type
# cotd = {3000: 'NONE', 1:'CROT'}
# for ticid in mg.objid:
#     if int(ticid) in targets:
#         numcot.append(1)
#     else:
#         numcot.append(3000)
# pt.plot_tsne(mg.feats, numcot, X=mg.tsne, output_dir=mg.featpath+'imgs/',
#              prefix=prefix, otypedict=cotd, alpha =1,
#              class_marker='x', class_ms=20, objid=mg.objid,
#              plot_insets=True, lcpath=mg.datapath+'clip/', inset_label=1)
 
for i in range(10):
    prefix = 'zoom_'+str(i)+'_'
    pt.plot_tsne(mg.feats, mg.clstr, X=mg.tsne, output_dir=mg.featpath+'imgs/',
                 prefix=prefix, objid=mg.objid, 
                 lcpath=mg.datapath+'clip/', zoom=True)
