from obspy import read
from resp_spec import *

sPeriod = np.array([0.01,0.02,0.022,0.025,0.029,0.03,0.032,0.035,0.036,
  0.04,0.042,0.044,0.045,0.046,0.048,0.05,0.055,0.06,0.065,0.067,0.07,
  0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.125,0.13,0.133,0.14,0.15,
  0.16,0.17,0.18,0.19,0.2,0.22,0.24,0.25,0.26,0.28,0.29,0.3,0.32,0.34,
  0.35,0.36,0.38,0.4,0.42,0.44,0.45,0.46,0.48,0.5,0.55,0.6,0.65,0.667,
  0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,
  2,2.2,2.4,2.5,2.6,2.8,3,3.2,3.4,3.5,3.6,3.8,4,4.2,4.4,4.6,4.8,5,7.5,
  10])

st = read('1994-01-17T12_30_12.010000Z.TS.SBC.BHZ.SAC',format = 'SAC')
# Be sure that the preprocesses such as detrend are applied to signal.
#st[0].detrend(type='simple')
#st[0].taper(max_percentage=0.05,type='hann')
PSA, PSV, SD = ins_resp(st[0].data*100, dt = st[0].stats.delta, periods = sPeriod, xi = 0.05)
plotting(PSA,PSV,SD,sPeriod,logplot = True,saving = 'show_save',title = str(st[0].stats.station + '.' + st[0].stats.channel))
