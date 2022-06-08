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
import matplotlib.pyplot as plt

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
for i in [10, 25,500, 1000]:
    mg.numclstr=i
    mg.generate_clusters()

pdb.set_trace()

# mg.load_tsne()
mg.load_gmm_clusters()

mg.load_true_otypes() # >> return numtot, otdict
# mg.generate_predicted_otypes()
# rec, fdr, pre, acc, cnts_true, cnts_pred = pt.evaluate_classifications(mg.cm)

# mg.numerize_otypes()
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

# # >> random
# for i in range(10):
#     prefix = 'zoom_'+str(i)+'_'
#     pt.plot_tsne(mg.feats, mg.clstr, X=mg.tsne, output_dir=mg.featpath+'imgs/',
#                  prefix=prefix, objid=mg.objid, 
#                  lcpath=mg.datapath+'clip/', zoom=True)

# >> cluster centers (mean of all positions of cluster members)
# foo = []
# cat = np.loadtxt(mg.metapath+'spoc/otypes_S1_26.txt', skiprows=2, delimiter=',',
#                  dtype='str')
# inter, comm1, comm2 = np.intersect1d(cat[:,0].astype('int'), mg.objid,
#                                      return_indices=True)
# cat = cat[:,1][comm1]

rad = []
bins = 40
unq_clstr = sorted(np.unique(mg.clstr))
cen_clstr = []
thr_clstr = []
label = []
for i in unq_clstr:
# for i in [23]:
    if i % 10 == 0:
        print(i)
    # prefix = 'zoom_clstr'+str(i)+'_'
    inds = np.nonzero(mg.clstr == unq_clstr[i])

    # >> find center
    # from scipy.stats import gaussian_kde
    # kde = gaussian_kde(mg.feats[inds].T)
    # density = kde(mg.feats[inds].T)
    # centr = mg.feats[inds][np.argmax(density)]
    # dist = np.sum((mg.feats[inds] - centr)**2, axis=1)

    # centr = np.mean(mg.feats[inds], axis=0)
    centr = np.median(mg.feats[inds], axis=0)
    centr_ind = np.argmin(np.sum((mg.feats[inds]-centr)**2, axis=1))
    # dist = np.sum((mg.feats[inds] - mg.feats[inds][centr_ind])**2, axis=1)
    dist = np.sum((mg.feats[inds] - centr)**2, axis=1)
    cen_clstr.append(centr)

    # >> threshold distance from center
    # thresh = np.median(dist) + 2* np.std(dist)
    thresh = np.median(dist) + np.std(dist)
    thr_clstr.append(thresh)
    
    # >> create histogram
    fig, ax = plt.subplots()
    ax.hist(dist, bins)
    ax.set_xlabel('Distance from cluster median')
    ax.set_ylabel('Number of cluster members')
    ax.axvline(thresh)

    # >> superimpose histograms of classified 
    ot, cnts = np.unique(mg.totype[inds], return_counts=True)
    score = []
    # ot_dist = []
    ot_true = [] 
    ot_cnts = []
    for j in range(len(ot)):
        if ot[j] != 'UNCLASSIFIED':
            inds_ot = np.nonzero(mg.totype[inds] == ot[j])
            ax.hist(dist[inds_ot], bins, alpha=0.5, label=ot[j])
            ot_true.append(ot[j])
            ot_cnts.append(np.count_nonzero(dist[inds_ot] < thresh))

    # pdb.set_trace()

    # >> method 1: label cluster (most abundant label within threshold)
    if len(ot_true) == 0:
        label.append('UNCLASSIFIED')
    elif len(ot_true) == 1:
        label.append(ot_true[0])
    else:
        if np.sort(ot_cnts)[-1] > np.sort(ot_cnts)[-2]+3*np.std(ot_cnts):
            label.append(ot_true[np.argmax(ot_cnts)])
        else:
            label.append('UNCLASSIFIED')

    rad.append(np.max(dist))

    ax.legend()
    ax.set_title(label[-1])
    fig.savefig(mg.featpath+'dist/hist_clstr_'+str(unq_clstr[i])+'.png')
    plt.close()
    # foo.append(mg.otdict[mg.numtot[centr_ind]])
    # pt.plot_tsne(mg.feats, mg.clstr, X=mg.tsne, output_dir=mg.featpath+'imgs/',
    #              prefix=prefix, objid=mg.objid, 
    #              lcpath=mg.datapath+'clip/', zoom=True, zoom_ind=centr_ind,
    #              numtot=mg.numtot, otdict=mg.otdict)

numpot = []
for i in range(len(mg.objid)):
    dist = np.sum((mg.feats[i] - cen_clstr[mg.clstr[i]])**2)
    if dist < thr_clstr[mg.clstr[i]]:
        class_ind = list(mg.otdict.values()).index(label[mg.clstr[i]])
        numpot.append(class_ind)
    else:
        unclassified_ind = list(mg.otdict.values()).index('UNCLASSIFIED')
        numpot.append(unclassified_ind)
numpot = np.array(numpot)

from sklearn.metrics import confusion_matrix
row_labels = []
inds = []
for i in np.unique(numpot):
    if mg.otdict[i] != 'UNCLASSIFIED':
        inds.extend(np.nonzero(mg.numtot == i)[0])
        row_labels.append(mg.otdict[i])
row_labels.append('UNCLASSIFIED')
inds = np.array(inds)
cm = confusion_matrix(mg.numtot[inds], numpot[inds])
pt.plot_confusion_matrix(cm, row_labels, row_labels, mg.featpath+'cm/',
                         'gmm'+str(numclstr)+'-',
                         figsize=(7,7))

rec, fdr, pre, acc, cnts_true, cnts_pred = pt.evaluate_classifications(cm)
