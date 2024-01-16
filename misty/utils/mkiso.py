from ..predict import GenModJax as GenMIST
from jax import jit
import jax.numpy as np

class iso(object):
    def __init__(self, *args, **kwargs):
        super(iso, self).__init__()
        self.args = args
        self.kwargs = kwargs

        # set NN path
        self.nnpath = kwargs.get('nnpath',None)
        
        # set type of NN
        self.NNtype = kwargs.get('NNtype','LinNet')

        # set if you want spot model to be applied in model call
        self.applyspot = kwargs.get('applyspot',False)
        
        # init the GenMist class
        GMIST = GenMIST.modpred(
            nnpath=self.nnpath,
            nntype=self.NNtype,
            normed=True,
            applyspot=self.applyspot)
        self.MISTpars = GMIST.modpararr

        # jit the function call so that it is faster
        self.genMISTfn = jit(GMIST.getMIST)
        
        # define eep array
        self.eepres = kwargs.get('eepsteps',808)
        # self.eeparr = np.arange(1,808+self.eepres,self.eepres)
        self.eeparr = np.linspace(10,606,self.eepres)
        
        # define mass array
        self.massres = kwargs.get('masssteps',300)
        # self.massarr = np.arange(0.25,5.0+self.massres,self.massres)
        # self.massarr = np.logspace(np.log10(0.25),np.log10(5.0),self.massres)
        self.massarr = np.linspace(0.4,4.0,self.massres)
        
    def __str__(self):
        return 'iso@{:#x}: {} {}'.format(id(self), self.args, self.kwargs)
    
    def geniso(self,age,feh,afe):
        
        # init the output dict
        out = {}
        for pp in self.MISTpars:
            out[pp] = []
        
        # cylce through masses, predict eep and age, find eep at input age,
        # then do prediction for the rest of the parameters
        
        for mm in self.massarr:
            age_i = []
            for eep in self.eeparr:
                mist_pred = self.genMISTfn(
                    eep=eep,
                    mass=mm,
                    feh=feh,
                    afe=afe,
                    verbose=False
                    )
                age_i.append(10.0**(float(mist_pred['log(Age)'])-9.0))
            
            # interpolate input age -> eep for this mass
            eep_t = np.interp(age,np.array(age_i),self.eeparr,left=np.nan,right=np.nan)
            
            if np.isfinite(eep_t):
                # make the prediction at this specific target eep
                mist_pred = self.genMISTfn(
                    eep=eep_t,
                    mass=mm,
                    feh=feh,
                    afe=afe,
                    verbose=False,
                )
                
                # write pars into out dict
                for pp in self.MISTpars:
                    out[pp].append(float(mist_pred[pp]))
            else:
                for pp in self.MISTpars:
                    out[pp].append(np.nan)
        return out
                