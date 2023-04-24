import h5py
import misty
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from datetime import datetime

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


    def selmod(self,inlabels,**kwargs):
        """
        Select nearest model to input labels

        inlabels -> one tuple or an array of tuples
                    (EEP,initial_mass,initial_[Fe/H],initial_[a/Fe])
        """

        norm = kwargs.get('norm',False)

        # force to be a list even if user submits only one set of labels
        if isinstance(inlabels[0],float):
            inlabels = [inlabels]

        # create arrays for outputs
        label_o = []
        #
        starmass_o = []
        logL_o = []
        logTeff_o = []
        logR_o = []
        logg_o = []
        logage_o = []
        FeHs_o = []
        aFes_o = []
        eep_o = []
        # Agewgt_o = []


        for li in inlabels:
            # split in labels
            EEP_i  = li[0]
            mass_i = li[1]
            FeH_i  = li[2]
            aFe_i  = li[3]

            # find nearest value to FeH and aFe
            FeH_ii   = self.FeHarr[np.argmin(np.abs(np.array(self.FeHarr)-FeH_i))]
            aFe_ii = self.aFearr[np.argmin(np.abs(np.array(self.aFearr)-aFe_i))]

            # pull set of models for nearest FeH and aFe
            mod_o = self.mist['{0:.2f}/{1:.2f}/0.40'.format(FeH_ii,aFe_ii)]
            eeparr  = mod_o['EEP']
            massarr = mod_o['initial_mass']

            # find index of nearest model
            mod_nearest = NearestNDInterpolator(
                np.array([eeparr,massarr]).T,range(0,len(eeparr))
                )((EEP_i,mass_i))

            # append outputs
            label_o.append(
                [mod_o['EEP'][mod_nearest],mod_o['initial_mass'][mod_nearest],FeH_ii,aFe_ii]
                )
            #
            if norm:
                starmass_o.append(self.normf(mod_o['star_mass'][mod_nearest],'star_mass'))
                logL_o.append(self.normf(mod_o['log_L'][mod_nearest],'log_L'))
                logTeff_o.append(self.normf(mod_o['log_Teff'][mod_nearest],'log_Teff'))
                logR_o.append(self.normf(mod_o['log_R'][mod_nearest],'log_R'))
                logg_o.append(self.normf(mod_o['log_g'][mod_nearest],'log_g'))
                logage_o.append(self.normf(mod_o['log_age'][mod_nearest],'log_age'))
                FeHs_o.append(self.normf(mod_o['[Fe/H]'][mod_nearest],'[Fe/H]'))
                aFes_o.append(self.normf(mod_o['[a/Fe]'][mod_nearest],'[a/Fe]'))
                eep_o.append(self.normf(mod_o['EEP'][mod_nearest],'EEP'))
                # Agewgt_o.append(self.normf(mod_o['Agewgt'][mod_nearest],'Agewgt'))
            else:
                starmass_o.append(mod_o['star_mass'][mod_nearest])
                logL_o.append(mod_o['log_L'][mod_nearest])
                logTeff_o.append(mod_o['log_Teff'][mod_nearest])
                logR_o.append(mod_o['log_R'][mod_nearest])
                logg_o.append(mod_o['log_g'][mod_nearest])
                logage_o.append(mod_o['log_age'][mod_nearest])
                FeHs_o.append(mod_o['[Fe/H]'][mod_nearest])
                aFes_o.append(mod_o['[a/Fe]'][mod_nearest])
                eep_o.append(mod_o['EEP'][mod_nearest])
                # Agewgt_o.append(mod_o['Agewgt'][mod_nearest])

        outdict = {}
        outdict['label_i']    = np.array(label_o)
        outdict['star_mass']  = np.array(starmass_o)
        outdict['log_L']      = np.array(logL_o)
        outdict['log_Teff']   = np.array(logTeff_o)
        outdict['log_R']      = np.array(logR_o)
        outdict['log_g']      = np.array(logg_o)
        outdict['log_age']    = np.array(logage_o)
        outdict['[Fe/H]']     = np.array(FeHs_o)
        outdict['[a/Fe]']     = np.array(aFes_o)
        outdict['EEP']        = np.array(eep_o)
        # outdict['Agewgt']      = np.array(Agewgt_o)

        return outdict

    def pullmod(self,num,**kwargs):
        """
        randomly draw models from mist grid
        """

        eeprange = kwargs.get('eep',None)
        if eeprange is None:
            eeprange = self.minmax['EEP']

        massrange = kwargs.get('mass',None)
        if massrange is None:
            massrange = self.minmax['initial_mass']

        fehrange = kwargs.get('feh',None)
        if fehrange is None:
            fehrange = self.minmax['initial_[Fe/H]']

        aferange = kwargs.get('afe',None)
        if aferange is None:
            aferange = self.minmax['initial_[a/Fe]']

        if 'excludelabels' in kwargs:
            excludelabels = kwargs['excludelabels'].T.tolist()
        else:
            excludelabels = []

        norm = kwargs.get('norm',False)

        eepprior = kwargs.get('eepprior',False)

        # create arrays for outputs
        label_o = []
        #
        starmass_o = []
        logL_o = []
        logTeff_o = []
        logR_o = []
        logg_o = []
        logage_o = []
        FeHs_o = []
        aFes_o = []
        eep_o = []
        # Agewgt_o = []

        starttimef = datetime.now()
        for ii in range(num):
            while True:
                # first randomly draw a [Fe/H]
                while True:
                    fehafe_i = np.random.choice(self.index,p=None)
                    FeH_i = float(fehafe_i.split('/')[0])
                    aFe_i = float(fehafe_i.split('/')[1])

                    # check to make sure FeH_i aFe_i are in user defined 
                    # [Fe/H] & [a/Fe] limits
                    if ((FeH_i >= fehrange[0]) & (FeH_i <= fehrange[1]) & 
                        (aFe_i >= aferange[0]) & (aFe_i <= aferange[1])):
                        break

                # pull the MIST tracks for FeH_i and aFe_i
                try:
                    mod_o = self.mist['{0:.2f}/{1:.2f}/0.40'.format(FeH_i,aFe_i)]
                except KeyError:
                    continue

                eeparr = self.eeparr[(
                    (self.eeparr >= eeprange[0]) &
                    (self.eeparr <= eeprange[1])
                    )]

                massarr = self.massarr[(
                    (self.massarr >= massrange[0]) &
                    (self.massarr <= massrange[1])
                    )]

                if eepprior:
                    # draw eep with weighting towards short lived phases 
                    # since that is where the isochrones change most
                    ueeparr = np.unique(eeparr)
                    peeparr = self.Peep(ueeparr)
                    peeparr = peeparr/peeparr.sum()
                    eep_i = np.random.choice(ueeparr,p=peeparr)
                else:
                    eep_i  = np.random.choice(np.unique(eeparr),p=None)
                mass_i = np.random.choice(np.unique(massarr),p=None)

                # assemble set of input labels 
                label_i = [eep_i,mass_i,FeH_i,aFe_i]

                # check to make sure labels are not in excludelabels
                if label_i in excludelabels:
                    continue

                # check to make sure label_i is not already in label_o
                if (label_i in label_o):
                    continue

                # select index of label_i
                ind = (mod_o['EEP'] == eep_i) & (mod_o['initial_mass'] == mass_i)

                # check if exact model is in mod_o, if not restart
                # This sometimes happens for low-mass models at very old ages
                if ind.sum() != 1:
                    continue

                try:
                    starmass_oo = mod_o['star_mass'][ind].item()
                    logL_oo     = mod_o['log_L'][ind].item()
                    logTeff_oo  = mod_o['log_Teff'][ind].item()
                    logR_oo     = mod_o['log_R'][ind].item()
                    logg_oo     = mod_o['log_g'][ind].item()
                    logage_oo   = mod_o['log_age'][ind].item()
                    FeHs_oo     = mod_o['[Fe/H]'][ind].item()
                    aFes_oo     = mod_o['[a/Fe]'][ind].item()
                    eep_oo      = mod_o['EEP'][ind].item()
                    # Agewgt_oo   = mod_o['Agewgt'][ind].item()
                except ValueError:
                    print(label_i)
                    print(ind.sum())
                    print(np.unique(massarr))
                    print(mod_o['star_mass'][ind])
                    raise

                # check to make sure log(Age) < np.log10(20E+9) and log(Age) > np.log10(1E+6)
                if (logage_oo > np.log10(20E+9)) | (logage_oo < np.log10(1E+6)):
                    continue

                # add label_i to label_o
                label_o.append(label_i)

                #
                if norm:
                    starmass_o.append(self.normf(starmass_oo,'star_mass'))
                    logL_o.append(self.normf(logL_oo,'log_L'))
                    logTeff_o.append(self.normf(logTeff_oo,'log_Teff'))
                    logR_o.append(self.normf(logR_oo,'log_R'))
                    logg_o.append(self.normf(logg_oo,'log_g'))
                    logage_o.append(self.normf(logage_oo,'log_age'))
                    FeHs_o.append(self.normf(FeHs_oo,'[Fe/H]'))
                    aFes_o.append(self.normf(aFes_oo,'[a/Fe]'))
                    eep_o.append(self.normf(eep_oo,'EEP'))
                    # Agewgt_o.append(self.normf(Agewgt_oo,'Agewgt'))
                else:
                    starmass_o.append(starmass_oo)
                    logL_o.append(logL_oo)
                    logTeff_o.append(logTeff_oo)
                    logR_o.append(logR_oo)
                    logg_o.append(logg_oo)
                    logage_o.append(logage_oo)
                    FeHs_o.append(FeHs_oo)
                    aFes_o.append(aFes_oo)
                    eep_o.append(eep_oo)
                    # Agewgt_o.append(Agewgt_oo)
                break


        outdict = {}
        outdict['label_i']    = np.array(label_o)
        outdict['star_mass']  = np.array(starmass_o).T
        outdict['log_L']      = np.array(logL_o).T
        outdict['log_Teff']   = np.array(logTeff_o).T
        outdict['log_R']      = np.array(logR_o).T
        outdict['log_g']      = np.array(logg_o).T
        outdict['log_age']    = np.array(logage_o).T
        outdict['[Fe/H]']     = np.array(FeHs_o).T
        outdict['[a/Fe]']     = np.array(aFes_o).T
        outdict['EEP']        = np.array(eep_o).T
        # outdict['Agewgt']      = np.array(Agewgt_o).T

        return outdict

    def Peep(self,x):
        # return 0.3*np.exp(-0.5*((x-450)/75)**2) + 0.7*np.exp(-0.5*((x-650)/150)**2)
        return 0.6*np.exp(-0.5*((x-425)/75)**2) + 0.4*np.exp(-0.5*((x-650)/100)**2)