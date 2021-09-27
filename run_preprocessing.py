"""
Created on 210919

run_preprocessing.py
Script to perform preprocessing on SPOC light curves, with the science focus of
clustering and identifying anomalies of complex rotators.
@author: Emma Chickles (emmachickles)

"""

from __init__ import *
import feat_eng as fe

# >> inputs
datapath     = '/scratch/data/tess/lcur/spoc/'
savepath     = '/scratch/echickle/'
# savepath = '/scratch/echickle/tmp/'
datatype     = 'SPOC'
mdumpcsv     = '/scratch/data/tess/meta/Table_of_momentum_dumps.csv'
sector       = 1
plot         = True
target       = 31850842 # >> for target_prep

# >> what to run?
sector_prep  = True  # >> preprocess a sector
target_prep  = False # >> preprocess a single target
diag_only    = False # >> produce diagnostic plots

# ------------------------------------------------------------------------------

# >> Initialize Mergen object
mg = mergen.mergen(datapath, savepath, datatype, sector=sector,
                   mdumpcsv=mdumpcsv)


# >> Perform quality and sigma-clip masking for a sector
if sector_prep:
    # mg.clean_data()
    # fe.sigma_clip_sector(mg, plot=plot)
    fe.compute_ls_pgram_sector(mg, plot=plot)

# >> Alternatively perform quality and sigma-clip masking for a single target
if target_prep:
    lcfile = str(target)+'.fits'
    # fe.sigma_clip_lc(mg, plot=plot)    

# >> Create some diagnostic plots for masking
if diag_only:
    dt.clean_sector_diag(datapath+'mask/sector-01/', mg.ensbpath, sector,
                         mdumpcsv)

# >> Compute LS Periodograms
if sector_prep:
    dt.compute_ls_pgram_sector(datapath, sector, plot=True,
                               savepath=mg.ensbpath)

