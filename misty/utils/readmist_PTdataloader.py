import h5py
import misty
import numpy as np
from datetime import datetime
from astropy.table import Table

class readmist(object):
    """docstring for readmist"""
    def __init__(self, **kwargs):
        super(readmist, self).__init__()

        # read in HDF5 file
        # determine path to atm files
        self.mistpath = kwargs.get('mistpath',None)

        if self.mistpath is None:
            self.mistpath = misty.__abspath__+'data/MIST/mist_default.h5'

        # read in HDF5 file
        mistfile = h5py.File(self.mistpath,'r')

        # read in index array
        self.index = [x.decode('utf-8') for x in list(mistfile['index'])]

        # create mist dictionary and determine all possible masses
        self.mist = {}
        self.massarr = []
        for ii in self.index:
            self.mist[ii] = np.array(mistfile[ii])
            self.massarr += list(np.unique(self.mist[ii]['initial_mass']))

        # create arrays for initial FeH and aFe
        self.FeHarr  = []
        self.aFearr  = []
        self.Vrotarr = []
        for ii in self.index:
            ii_s = ii.split('/')
            self.FeHarr.append( float(ii_s[0]))
            self.aFearr.append( float(ii_s[1]))
            self.Vrotarr.append(float(ii_s[2]))
        self.FeHarr  = np.unique(self.FeHarr)
        self.aFearr  = np.unique(self.aFearr)
        self.Vrotarr = np.unique(self.Vrotarr)
        self.FeHarr.sort()
        self.aFearr.sort()
        self.Vrotarr.sort()

        # determine arrays for all possible eeps and masses
        self.massarr = np.unique(self.massarr)
        self.eeparr = np.arange(1,808)

        # figure out columns used in models
        # self.columns = list(np.array(HDF5file[self.index[0]]).dtype.names)
        # default set of columns
        self.columns = ([
            'EEP',
            'initial_[Fe/H]',
            'initial_[a/Fe]',
            'initial_mass',
            'star_mass',
            'log_L',
            'log_Teff',
            'log_R',
            'log_g',
            'log_age',
            '[Fe/H]',
            '[a/Fe]',
            'Agewgt',
            ])

        # need min-max values for each column used in models
        # self.minmax = {x:[np.inf,-np.inf] for x in self.colnames}
        # default set of min-max values
        self.minmax = ({
            'EEP':[1,808],
            'initial_mass':[0.25,10.0],
            'initial_[Fe/H]':[-4,0.5],
            'initial_[a/Fe]':[-0.2,0.6],
            'star_mass':[0.25,10.0],
            'log_L':[-3.0,5.0],
            'log_Teff':[np.log10(2500.0),np.log10(50000.0)],
            'log_R':[-2.0,4.0],
            'log_g':[-1,5.5],
            'log_age':[6.0,np.log10(20E+9)],
            '[Fe/H]':[-4.0,0.5],
            '[a/Fe]':[-0.2,0.6],
            'Agewgt':[0,0.05],
            })

    def normf(self,inarr,label):
        return ((inarr-self.minmax[label][0])/(self.minmax[label][1]-self.minmax[label][0])) - 0.5

