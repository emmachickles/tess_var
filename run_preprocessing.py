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

# >> Initialize Mergen object
mg = mergen.mergen(datapath, savepath, datatype, sector=sector,
                   mdumpcsv=mdumpcsv)

# >> Perform quality and sigma-clip masking
# mg.clean_data()

# >> Let's create some diagnostic plots 
dt.clean_sector_diag(datapath+'mask/sector-01/', mg.ensbpath, sector, mdumpcsv)

# lc = 'tess2018206045859-s0001-0000000025132999-0120-s_lc.fits'
# lcfile = datapath+'sector-01/'+lc


# for lc in os.listdir(datapath+'sector-01')[:10]:
#     lcfile = datapath+'sector-01/'+lc
#     dt.general_preprocessing(lcfile, mdumpcsv)
