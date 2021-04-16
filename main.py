"""
210402 - Emma Chickles
"""

from __init__ import *

import feat_eng as fe

datapath = '/nfs/blender/data/tdaylan/data/'
savepath = '/nfs/blender/data/tdaylan/Mergen_Run_2/'
datatype = 'SPOC'
mdumpcsv = datapath + 'Table_of_momentum_dumps.csv'
sector   = 1
n_freq   = 400000
n_terms  = 1
n_bins   = 100

obj = mg.mergen(datapath, savepath, datatype, mdumpcsv, sector=sector)
obj.load_lightcurves_local()

targ = [25078924, 31655792, 260162199, 144194304]
# targ = [31655792, 260162199, 144194304]
for t in targ:
    ind = np.nonzero(obj.ticid==t)
    y = obj.intensities[ind][0]

    prefix = 'TIC'+str(t)+'-nfreq'+str(n_freq)+'-nterms-'+str(n_terms)+'-'
    fe.calc_phase_curve(obj.times, y, output_dir='/nfs/ger/home/echickle/',
                        prefix=prefix, plot=True, n_freq=n_freq,
                        n_terms=n_terms, n_bins=n_bins)

pdb.set_trace()
fe.create_phase_curve_feats(obj.times, obj.intensities, sector=obj.sector,
                            output_dir=obj.savepath, n_freq=n_freq,
                            n_terms=n_terms, n_bins=n_bins)

