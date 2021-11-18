"""
Created on 211023

run_matching.py
Script to obtaining classifications from ASAS-SN and Simbad.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *
import cross_match as cm

# -- inputs --------------------------------------------------------------------

datapath = '/scratch/data/tess/lcur/spoc/'
savepath = '/scratch/echickle/'
metapath = '/scratch/data/tess/meta/'
datatype = 'SPOC'

# -- main ----------------------------------------------------------------------

mg = mergen(datapath, savepath, datatype, metapath=metapath)
# mg.initiate_meta()
# cm.query_asas_sn(mg.metapath, mg.savepath)
# cm.query_simbad(mg.metapath, mg.savepath)
cm.clean_simbad(mg.metapath) 
# cm.clean_asassn(mg.metapath)
cm.write_true_label_txt(mg.metapath, catalogs=['simbad', 'asassn'])
