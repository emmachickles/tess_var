"""
Created on 211023

run_matching.py
Script to obtaining classifications from ASAS-SN and Simbad.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import cross_match as cm
import feat_eng as fe
import numpy as np

# -- inputs --------------------------------------------------------------------

datapath = '/scratch/data/tess/lcur/spoc/'
savepath = '/scratch/echickle/'
metapath = '/scratch/data/tess/meta/'
datatype = 'SPOC'
featgen  = 'DAE'

# -- main ----------------------------------------------------------------------

mg = mergen(datapath, savepath, datatype, metapath=metapath, featgen=featgen)
# mg.initiate_meta()
# cm.query_asas_sn(mg.metapath, mg.savepath)
# cm.query_simbad(mg.metapath, mg.savepath)
cm.clean_simbad(mg.metapath) 
# cm.clean_asassn(mg.metapath)
cm.write_true_label_txt(mg.metapath, catalogs=['simbad', 'asassn'])
fe.load_lspgram_fnames(mg)
mg.load_true_otypes()
otypes_all = np.unique(mg.totype)
np.savetxt(mg.savepath+'otypes_all.txt', otypes_all, delimiter=',', fmt='%s')
print(otypes_all[::-1])
print('Number of classes: '+str(len(otypes_all)))

# >> check if there's unaccounted for variability types
var_d = cm.make_variability_tree()
subclasses = []
for i in var_d.keys():
    subclasses.extend(var_d[i])
print(subclasses)
new = []
for i in range(len(otypes_all)):
    otype_list = otypes_all[i].split('|')
    for j in range(len(otype_list)):
        if otype_list[j] not in subclasses:
            new.append(otype_list[j])
print(np.unique(new))
print('Unaccounted for:'+str(len(np.unique(new))))



# >> 1) grep /scratch/data/tess/meta/spoc/cat/*_cln.txt
# >> bugs

