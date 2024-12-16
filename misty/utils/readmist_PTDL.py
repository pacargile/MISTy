import h5py
import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from torch.utils.data import Dataset

class readmist(Dataset):
    """docstring for readmist"""
    def __init__(self, **kwargs):
        super(readmist, self).__init__()

        self.verbose = kwargs.get('verbose',False)

        # read in HDF5 file
        # determine path to atm files
        self.mistpath = kwargs.get('mistpath',None)

        if self.mistpath is None:
            self.mistpath = "./mist_default.h5"

        if self.verbose:
            print(f'... MIST path: {self.mistpath}')

        # set training boolean, if false then assume test set
        self.datatype = kwargs.get('type','train')
        if self.verbose:
            print(f'... Data Set Type: {self.datatype}')

        # define a RNG with a set seed, make sure train and test are 
        # shuffled the same way
        if (self.datatype == 'train') | (self.datatype == 'test'):
            rng = np.random.default_rng(0)
        else:
            rng = np.random.default_rng(42)

        # set if to do the normalization or not
        self.norm = kwargs.get('norm',True)
        if self.verbose:
            print(f'... Normalize: {self.norm}')
        
        # set if user wants to return a pytorch tensor or a numpy array
        self.returntorch = kwargs.get('returntorch',True)
        if self.verbose:
            print(f'... Return Torch Tensor: {self.returntorch}')

        # set train/test percentage (percentage of MIST that is used for training)
        self.trainper = kwargs.get('trainpercentage',0.9)
        if self.verbose:
            print(f'... Training/Test Percentage: {self.trainper}')

        # check for user defined ranges for atm models
        # defaultparrange = ({
        #     'EEP':[0.0,1000000.0],
        #     'initial_mass':[-1.0,10000.0],
        #     'initial_[Fe/H]':[-10.0,10.0],
        #     'initial_[a/Fe]':[-10.0,10.0],
        #     })
        defaultparrange = None
        self.parrange = kwargs.get('parrange',defaultparrange)

        default_label_i = ([
            'EEP',
            'initial_mass',
            'initial_[Fe/H]',
            'initial_[a/Fe]',
            ])
        
        self.label_i = kwargs.get('label_i',default_label_i)

        if self.verbose:
            print('... Input Labels:')
            print(f'{self.label_i}')

        # read in HDF5 file
        mistfile = h5py.File(self.mistpath,'r')

        # read in index array
        self.index = [x.decode('utf-8') for x in list(mistfile['index'])]
        self.index_labels = mistfile.attrs['indexlabel'].split('|')
        self.index_fmt = mistfile.attrs['indexfmt'].split('|')

        # create mist dictionary and determine all possible masses
        self.mist = {}

        for indexstr in self.index:
            # parse index string
            indexstr_s = indexstr.split('/')

            # pull mist models for this index string
            self.mist[indexstr] = np.array(mistfile[indexstr])
            nrows = len(self.mist[indexstr])
            
            # add index string parameters to dict
            inpars = {x:float(y)*np.ones(nrows,dtype=float) for (x,y) in zip(self.index_labels,indexstr_s)}
            # add other input labels to dict
            for il in self.label_i:
                if il not in list(inpars.keys()):
                    inpars[il] = self.mist[indexstr][il]            

            # add these columns to output mist table
            addcols = [x for x in inpars.keys() if x not in self.mist[indexstr].dtype.names]            
            for kk in addcols:
                self.mist[indexstr] = rfn.append_fields(self.mist[indexstr],kk,np.array(inpars[kk],dtype=float),usemask=False)

            # create conditional to parse down input labels to user
            # defined range

            cond = np.ones(nrows,dtype=bool)
            if self.parrange is not None:
                for kk in self.parrange.keys():
                    if kk in inpars.keys():
                        cond *= (inpars[kk] >= self.parrange[kk][0]) & (inpars[kk] <= self.parrange[kk][1])
                    if kk in self.mist[indexstr].dtype.names:
                        cond *= (self.mist[indexstr][kk] >= self.parrange[kk][0]) & (self.mist[indexstr][kk] <= self.parrange[kk][1])
                                        
            # check to make sure the parrange didn't remove all rows, if so, skip to the next index
            if cond.sum() == 0:
                continue
            
            parlist_i = []
            parlist_lab = []
            for kk in inpars.keys():
                parlist_lab.append(kk)
                parlist_i.append(inpars[kk][cond])

            pararr = np.array(parlist_i).T

            if indexstr == self.index[0]:
                self.allpars = pararr
                self.allpars_labels = parlist_lab
            else:
                self.allpars = np.vstack([self.allpars,pararr])

            # determine column names
            if indexstr == self.index[0]:
                self.columns = np.array(self.mist[indexstr].dtype.names)
                self.normfactor = {}

            # update median and min/max values for normalization
            for kk in self.columns:
                med = np.median(self.mist[indexstr][kk][cond])
                minmax = [min(self.mist[indexstr][kk][cond]),max(self.mist[indexstr][kk][cond])]
                
                # catch cases where min == max 
                # these shouldn't be used in training since there is no variance
                # but just to keep from throwing errors, make max-min = 1.0
                if minmax[0] == minmax[1]:
                    minmax = [med,med]
                
                if indexstr == self.index[0]:
                    self.normfactor[kk] = [med,minmax[0],minmax[1]]
                else:
                    self.normfactor[kk] = ([
                        np.median([self.normfactor[kk][0],med]),
                        min([self.normfactor[kk][1],minmax[0]]),
                        max([self.normfactor[kk][2],minmax[1]]),
                        ])

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

    def normf(self,inarr,label):        
        return 1.0 + (inarr-self.normfactor[label][0])/(self.normfactor[label][2]-self.normfactor[label][1]) 

    def unnormf(self,inarr,label):
        return ((inarr-1.0)*(self.normfactor[label][2]-self.normfactor[label][1])) + self.normfactor[label][0]

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
        pardict = {kk:parind[ii] for ii,kk in enumerate(self.allpars_labels)}
        
        # construct the key that pulls the correct MIST pars
        parkey = '/'.join([('{0:'+f'{ll}'+'}').format(pardict[kk]) for kk,ll in zip(self.index_labels,self.index_fmt)])
        mist_i = self.mist[parkey]
        
        # build cond for label_in that are not in index_labels
        cond = np.ones(len(mist_i),dtype=bool)
        for kk in self.label_i:
            if kk not in self.index_labels:
                cond *= mist_i[kk] == pardict[kk]
        
        # apply cond to mist_i
        mist_ii = mist_i[cond]
        
        # build a pararr for all columns in mist_i[cond]
        if self.norm:
            pararr = np.array([self.normf(mist_ii[x][0],x) for x in self.columns],dtype=np.float32)
        else:
            pararr = np.array([mist_ii[x][0] for x in self.columns],dtype=np.float32)
                

        if self.returntorch:
            pararr = torch.from_numpy(pararr)
            
        return pararr

