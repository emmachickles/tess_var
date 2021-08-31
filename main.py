"""
210402 - Emma Chickles
"""

from __init__ import *

import feat_eng as fe

# -- inputs --------------------------------------------------------------------
datapath = '/nfs/blender/data/tdaylan/data/'
savepath = '/nfs/blender/data/tdaylan/Mergen_Run_2/'
output_dir='./'
datatype = 'SPOC'
sector   = 1
n_freq   = 50000
n_freq0  = 10000
n_terms  = 2
n_bins   = 100
kernel_size = 15
thresh_min = 1e-2
plot = True
report_time = True

# -- initialize Mergen object --------------------------------------------------
mg = mergen.mergen(datapath, savepath, datatype, sector=sector)
mg.load_lightcurves_local()
mg.load_true_otypes()
mg.numerize_true_otypes()

# >> periodic examples
# targ = [25078924, 31655792, 260162199, 144194304]
# targ = [31655792,32200310,53842685,117515123,126637765,141268467,141606986,144194304,147083089,161262226,200645730,200655763,206537272,206555954,211412634,231629787,235000060,238170857,259901124,260162199,261089147,261259291,263078276,267043786,279955276,294273900,348717439,355544723,369218857,412063998, 419635446, 441105436]
# targ = [141268467]
# targ = [260162199, 144194304]

# >> non-periodic examples
targ = [24695725, 25132999, 25133286, 25299762, 25300088, 29285387]

# >> all
# targ = mg.ticid

targ = np.array(targ).astype('int')

# -- find periodic objects -----------------------------------------------------

ticid_periodic = fe.find_periodic_obj(mg.ticid, targ, mg.intensities, mg.times,
                                      mg.sector, output_dir, n_freq0=n_freq0,
                                      report_time=report_time, plot=plot,
                                      thresh_min=thresh_min)

# -- make phase curve features -------------------------------------------------

_, srtinds, _ = np.intersect1d(mg.ticid, ticid_periodic, return_indices=True)
flux_periodic = mg.intensities[srtinds]
tic_periodic = mg.ticid[srtinds]

fe.create_phase_curve_feats(mg.times, flux_periodic, tic_periodic,
                            sector=mg.sector, plot=plot,
                            output_dir=output_dir, n_freq=n_freq,
                            n_terms=n_terms, n_bins=n_bins,
                            n_freq0=n_freq0, report_time=report_time)
