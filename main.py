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
n_freq   = 100000
n_freq0  = 20000
n_terms  = 2
n_bins   = 100
kernel_size = 15
frac_max = 1e-2

obj = mg.mergen(datapath, savepath, datatype, mdumpcsv, sector=sector)
obj.load_lightcurves_local()

# targ = [25078924, 31655792, 260162199, 144194304]
targ = [31655792,32200310,53842685,117515123,126637765,141268467,141606986,144194304,147083089,161262226,200645730,200655763,206537272,206555954,211412634,231629787,235000060,238170857,259901124,260162199,261089147,261259291,263078276,267043786,279955276,294273900,348717439,355544723,369218857,412063998, 419635446, 441105436]
# targ = [141268467]
# targ = [260162199, 144194304]

# >> non-periodic
# targ = [24695725, 25132999, 25133286, 25299762, 25300088, 29285387]
for t in targ:
    ind = np.nonzero(obj.ticid==t)
    y = obj.intensities[ind][0]

    prefix = 'TIC'+str(t)+'-nfreq'+str(n_freq)+'-nterms-'+str(n_terms)+'-'
    fe.calc_phase_curve(obj.times, y, output_dir='/nfs/ger/home/echickle/',
                        prefix=prefix, plot=True, n_freq=n_freq,
                        n_terms=n_terms, n_bins=n_bins,
                        kernel_size=kernel_size, frac_max=frac_max)

pdb.set_trace()
fe.create_phase_curve_feats(obj.times, obj.intensities, sector=obj.sector,
                            output_dir=obj.savepath, n_freq=n_freq,
                            n_terms=n_terms, n_bins=n_bins,
                            n_freq0=n_freq0)

