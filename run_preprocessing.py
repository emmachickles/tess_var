"""
Created on 210919

run_preprocessing.py
Script to perform preprocessing on SPOC light curves, with the science focus of
clustering and identifying anomalies of complex rotators.
@author: Emma Chickles (emmachickles)
"""

from __init__ import *

# >> inputs
datapath     = '/scratch/data/tess/lcur/spoc/'
savepath     = '/scratch/echickle/'
datatype     = 'SPOC'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
sector       = 1
plot         = True
target       = 31850842 # >> for target_prep

# >> what to run?
sector_prep  = False  # >> preprocess a sector
target_prep  = True # >> preprocess a single target
diag_only    = False # >> produce diagnostic plots

# ------------------------------------------------------------------------------

# >> Initialize Mergen object
mg = mergen.mergen(datapath, savepath, datatype, sector=sector,
                   mdumpcsv=mdumpcsv)


# >> Perform quality and sigma-clip masking for a sector
if sector_prep:
    mg.clean_data()

# >> Alternatively perform quality and sigma-clip masking for a single target
if target_prep:
    rawspath = datapath+'raws/sector-%02d/'%sector
    maskpath = datapath+'mask/sector-%02d/'%sector
    lcfile = [lc for lc in os.listdir(rawspath) if str(target) in lc][0]
             # 'tess2018206045859-s0001-0000000389051009-0120-s_lc.fits'
    lcfile = rawspath+lcfile
    dt.clean_lc(lcfile, maskpath, mdumpcsv,
                plot=True, savepath=mg.ensbpath+'mask/')

# >> Create some diagnostic plots for masking
if diag_only:
    dt.clean_sector_diag(datapath+'mask/sector-01/', mg.ensbpath, sector,
                         mdumpcsv)

# >> Compute LS Periodograms
if sector_prep:
    dt.compute_ls_pgram_sector(datapath, sector, plot=True,
                               savepath=mg.ensbpath)

