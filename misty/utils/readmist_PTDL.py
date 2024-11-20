import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class readmist(Dataset):
    """docstring for readmist"""
    def __init__(self, **kwargs):
        super(readmist, self).__init__()

        # read in HDF5 file
        # determine path to atm files
        self.mistpath = kwargs.get('mistpath',None)

        if self.mistpath is None:
            self.mistpath = "./mist_default.h5"

        # set training boolean, if false then assume test set
        self.datatype = kwargs.get('type','train')

        # define a RNG with a set seed, make sure train and test are 
        # shuffled the same way
        if (self.datatype == 'train') | (self.datatype == 'test'):
            rng = np.random.default_rng(0)
        else:
            rng = np.random.default_rng(42)

        # set if to do the normalization or not
        self.norm = kwargs.get('norm',True)
        
        # set if user wants to return a pytorch tensor or a numpy array
        self.returntorch = kwargs.get('returntorch',True)

        # set train/test percentage (percentage of MIST that is used for training)
        self.trainper = kwargs.get('trainpercentage',0.9)

        # check for user defined ranges for atm models
        self.eeprange  = kwargs.get('eep',[0.0,1000000.0])
        self.massrange = kwargs.get('mass',[-1.0,10000.0])
        self.fehrange  = kwargs.get('feh',[-10.0,10.0])
        self.aferange  = kwargs.get('afe',[-10.0,10.0])

        # read in HDF5 file
        mistfile = h5py.File(self.mistpath,'r')

        # read in index array
        self.index = [x.decode('utf-8') for x in list(mistfile['index'])]

        # create mist dictionary and determine all possible masses
        self.mist = {}
        self.eeparr  = []
        self.massarr = []
        self.FeHarr  = []
        self.aFearr  = []
        self.Vrotarr = []

        for ii in self.index:
            ii_s = ii.split('/')            
            self.Vrotarr.append(float(ii_s[2]))

            self.mist[ii] = np.array(mistfile[ii])

            eep_i  = self.mist[ii]['EEP']
            mass_i = self.mist[ii]['initial_mass']
            feh_i  = self.mist[ii]['initial_[Fe/H]']
            afe_i  = self.mist[ii]['initial_[a/Fe]']

            parlist_i = []
            cond = (
                (eep_i >= self.eeprange[0]) & (eep_i <= self.eeprange[1]) &
                (mass_i >= self.massrange[0]) & (mass_i <= self.massrange[1]) &
                (feh_i >= self.fehrange[0]) & (feh_i <= self.fehrange[1]) &
                (afe_i >= self.aferange[0]) & (afe_i <= self.aferange[1])
            )

            parlist_i.append(eep_i[cond])
            parlist_i.append(mass_i[cond])
            parlist_i.append(feh_i[cond])
            parlist_i.append(afe_i[cond])

            pararr = np.array(parlist_i).T

            if ii == self.index[0]:
                self.allpars = pararr
            else:
                self.allpars = np.vstack([self.allpars,pararr])

            self.eeparr  += list(np.unique(self.mist[ii]['EEP']))
            self.massarr += list(np.unique(self.mist[ii]['initial_mass']))
            self.FeHarr += list(np.unique(self.mist[ii]['initial_[Fe/H]']))
            self.aFearr += list(np.unique(self.mist[ii]['initial_[a/Fe]']))

        # shuffle allpars
        rng.shuffle(self.allpars)

        if (self.datatype == 'train') | (self.datatype == 'valid'):
            stopind = int(np.rint(self.trainper * self.allpars.shape[0]))
            self.allpars = self.allpars[:stopind]
        else:
            startind = int(np.rint((1.0-self.trainper) * self.allpars.shape[0]))
            self.allpars = self.allpars[-startind:]

        # determine how many rows of data are included
        self.datalen = self.allpars.shape[0]

        # figure out general unique arrays
        self.eeparr = np.unique(self.eeparr)
        self.massarr = np.unique(self.massarr)
        self.FeHarr  = np.unique(self.FeHarr)
        self.aFearr  = np.unique(self.aFearr)
        self.Vrotarr = np.unique(self.Vrotarr)
        self.eeparr.sort()
        self.massarr.sort()
        self.FeHarr.sort()
        self.aFearr.sort()
        self.Vrotarr.sort()

        # figure out columns used in models
        # self.columns = list(np.array(HDF5file[self.index[0]]).dtype.names)
        # default set of columns
        self.columns = ([
            'EEP',
            'initial_mass',
            'initial_[Fe/H]',
            'initial_[a/Fe]',
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
            'initial_mass':[0.25,1.5],
            'initial_[Fe/H]':[-4,0.5],
            'initial_[a/Fe]':[-0.2,0.6],
            'star_mass':[0.25,1.5],
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

    def unnormf(self,inarr,label):
        return ((inarr + 0.5) * (self.minmax[label][1]-self.minmax[label][0])) + self.minmax[label][0]

    def __len__(self):
        """
        Return the total number of MIST rows        

        Returns:
            float : Total rows in MIST table
        """
        return self.datalen
    
    def __getitem__(self, idx):
        """
        Return a draw from the MIST table

        Args:
            idx (integer): index integer for row to draw
        """

        # select which set of parameters
        parind = self.allpars[idx]
        
        eep_i  = parind[0]
        mass_i = parind[1]
        feh_i = parind[2]
        afe_i = parind[3]
        vrot_i = self.Vrotarr[0]
        
        # construct the key that pulls the correct MIST pars
        parkey = f'{feh_i:.2f}/{afe_i:.2f}/{vrot_i:.2f}'
        mist_i = self.mist[parkey]
        
        cond = (mist_i['EEP'] == eep_i) & (mist_i['initial_mass'] == mass_i)
        mist_ii = mist_i[cond]        

        if self.norm:
            pararr = np.array([self.normf(mist_ii[x][0],x) for x in self.columns],dtype=np.float32)
        else:
            pararr = np.array([mist_ii[x][0] for x in self.columns],dtype=np.float32)
        

        if self.returntorch:
            pararr = torch.from_numpy(pararr)
            
        return pararr

