import sys
sys.path.insert(0, '../mergen/mergen')
import data_utils as du
from astropy.timeseries import TimeSeries
from astropy.time import Time
import numpy as np
import feat_eng as fe

data_dir = '/nfs/blender/data/tdaylan/data/'
output_dir = '../'
sector=1
n_freq=[1000, 5000]
n_terms=[2]
targ = [25078924, 31655792, 260162199, 144194304]

flux, time, ticid, target_info = \
    du.load_data_from_metafiles(data_dir, sector, output_dir=output_dir)

for freq in n_freq:
    for terms in n_terms:
        for t in targ:
            ind = np.nonzero(ticid==t)
            y = flux[ind][0]

            prefix = 'TIC'+str(t)+'-nfreq'+str(freq)+'-nterms-'+str(terms)+'-'
            fe.phase_curve(time,y,output_dir=output_dir, prefix=prefix, plot=True,
                           n_freq=freq, n_terms=terms)

    

# t = Time(time, format='jd')
# ts = TimeSeries(time=t, data={'flux': flux})
