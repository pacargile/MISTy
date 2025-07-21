import jax
import jax.numpy as jnp
from jax import jit,vmap,lax
import jax.random as jr

from flowjax.distributions import Normal,Transformed
from flowjax.flows import block_neural_autoregressive_flow,masked_autoregressive_flow
from flowjax.train import fit_to_data
import flowjax.bijections as bij
import equinox as eqx
import optax

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from glob import glob
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import scoreatpercentile

from astropy.table import Table
from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)

from misty.predict import GenModJax as GenMIST

class IMF_Prior(object):
    def __init__(self,low=0.25, high=3.0, alpha_low=1.3, alpha_high=2.3, mass_break=0.5, validate_args=None):
        """
        Apply a Kroupa-like broken IMF prior over the provided initial mass grid.
        Parameters
        ----------

        alpha_low : float, optional
            Power-law slope for the low-mass component of the IMF.
            Default is `1.3`.
        alpha_high : float, optional
            Power-law slope for the high-mass component of the IMF.
            Default is `2.3`.
        mass_break : float, optional
            The mass where we transition from `alpha_low` to `alpha_high`.
            Default is `0.5`.
        """
                
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.mass_break = mass_break

        # Compute normalization.
        norm_low = mass_break ** (1. - alpha_low) / (alpha_high - 1.)
        norm_high = 0.08 ** (1. - alpha_low) / (alpha_low - 1.)  # H-burning limit
        norm_high -= mass_break ** (1. - alpha_low) / (alpha_low - 1.)
        norm = norm_low + norm_high
        self.lognorm = jnp.log(norm)

        # define mass array to draw from
        self.massarr = jnp.linspace(low,high,100000)

        # determine prob map for mass array
        self.p = 10.0**vmap(self.log_prob,in_axes=(0))(self.massarr)
        

    def log_prob(self, mass):
        """
        mgrid : `~numpy.ndarray` of shape (Ngrid)
            Grid of initial mass (solar units) the IMF will be evaluated over.
        Returns
        -------
        lnprior : `~numpy.ndarray` of shape (Ngrid)
            The corresponding unnormalized ln(prior).
        """

        def lnprior_high(mass):
            return (-self.alpha_high * jnp.log(mass) 
                + (self.alpha_high - self.alpha_low) * jnp.log(self.mass_break))
        def lnprior_low(mass):
            return -self.alpha_low * jnp.log(mass)

        lnprior = lax.cond(mass > self.mass_break,lnprior_high,lnprior_low,mass)

        return lnprior - self.lognorm

    def sample(self,n=1,**kwargs):
        rng_seed = int.from_bytes(os.urandom(4),byteorder='big')
        key, subkey = jr.split(jr.key(rng_seed))        
        
        return jr.choice(subkey,self.massarr,shape=(n,),replace=False,p=self.p)

# Custom dataset class for building MIST samples
class MIST_dataloader(object):
    def __init__(self, *args, **kwargs):
        # self.rng_seed = kwargs.get('seed',1)
        self.rng_seed = int.from_bytes(os.urandom(4),byteorder='big')
        self.rng_key = jr.key(self.rng_seed)

        default_nnpath =  '/Users/pcargile/Astro/MISTy/train/v256_eep_v3/mistyNN_2.3_v256_v0.h5'
        self.nnpath = kwargs.get('nnpath',default_nnpath)

        # init MISTy
        GMIST = GenMIST.modpred(
                nnpath=self.nnpath,
                nntype='LinNet',
                normed=True,
                applyspot=False)
        self.genMISTfn = jit(GMIST.getMIST)        

        # build vmap'ed mist predict over eep
        self.mist_eepvmap = vmap(jit(self.samplemist), in_axes=(0,None,None,None))
        self.eeparr = jnp.linspace(100,808,1000)

        # vmap iso construction
        self.isocon = vmap(jit(self.compmistmass), in_axes=(0,None,None,None))

        self.minmass = kwargs.get('minmass',0.25)
        self.maxmass = kwargs.get('maxmass',5.0)

        self.minage = kwargs.get('minage',0.01)
        self.maxage = kwargs.get('maxage',14.0)
        
        # # init IMF
        # self.imf = IMF_Prior(low=kwargs.get('low_mass',self.minmass),high=kwargs.get('high_mass',self.maxmass))

    def compmistmass(self,mass,age,feh,afe):
        mist_eeptrack = self.mist_eepvmap(self.eeparr,mass,feh,afe)
        eep_t = jnp.interp(age,mist_eeptrack['Age'],mist_eeptrack['EEP'],left=jnp.nan,right=jnp.nan)
        pars_t = self.samplemist(eep_t,mass,feh,afe)
        return pars_t

        
    def samplemist(self,eep_i,mass_i,feh_i,afe_i):
        mistout =  self.genMISTfn(
            eep=eep_i,
            mass=mass_i,
            feh=feh_i,
            afe=afe_i,
            verbose=False)

        # change log(Teff) into Teff
        mistout['Teff'] = 10.0**mistout['log(Teff)']
        del mistout['log(Teff)']
        
        # change log(Age) into Age in Gyr
        mistout['Age'] = 10.0**(mistout['log(Age)']-9.0)
        del mistout['log(Age)']
        
        return mistout

    def buildssp(self,age=None,feh=0.0,afe=0.0,nsamp=1000,loggboost=False,ageboost=False):
        ii = 0
        # with tqdm(total=nsamp) as pbar:
        while True:
            self.rng_key,rng_key_tmp = jr.split(self.rng_key)
            if age != None:
                massarr = jr.beta(rng_key_tmp,1.0,1.0,shape=(1000,)) * (self.maxmass-self.minmass) + self.minmass
                singleiso = self.isocon(massarr,age,feh,afe)
            else:
                massarr = jr.beta(rng_key_tmp,1.0,1.0,shape=(50,)) * (self.maxmass-self.minmass) + self.minmass

                if ageboost:
                    agearr = jr.beta(rng_key_tmp,1.0,1.0,shape=(50,)) * (self.maxage-self.minage) + self.minage
                else:
                    agearr = jr.beta(rng_key_tmp,1.0,1.0,shape=(50,)) * (self.maxage-self.minage) + self.minage

                for jj,age_i in enumerate(agearr):
                    singleiso_i = self.isocon(massarr,age_i,feh,afe)
                    
                    if jj == 0:
                        singleiso = singleiso_i.copy()
                    else:
                        for kk in singleiso_i.keys():
                            singleiso[kk] = jnp.concatenate([singleiso[kk],singleiso_i[kk]])

            # filter nan's
            cond = jnp.isfinite(singleiso['Teff'])

            for kk in singleiso.keys():
                singleiso[kk] = singleiso[kk][cond]
                    
            # add jitter on parameters that go into The Payne
            singleiso['Teff']   = singleiso['Teff']   + 25.0 * jr.truncated_normal(rng_key_tmp,-5.0,5.0,shape=singleiso['Teff'].shape)
            singleiso['log(g)'] = singleiso['log(g)'] + 0.01 * jr.truncated_normal(rng_key_tmp,-5.0,5.0,shape=singleiso['log(g)'].shape)
            singleiso['[Fe/H]'] = singleiso['[Fe/H]'] + 0.01 * jr.truncated_normal(rng_key_tmp,-5.0,5.0,shape=singleiso['[Fe/H]'].shape)
            singleiso['[a/Fe]'] = singleiso['[a/Fe]'] + 0.01 * jr.truncated_normal(rng_key_tmp,-5.0,5.0,shape=singleiso['[a/Fe]'].shape)
            
            if ii == 0:
                ssp = {kk:jnp.array([]) for kk in singleiso.keys()}
            
            for kk in singleiso.keys():
                ssp[kk] = jnp.concatenate([ssp[kk],singleiso[kk]])
            
            starnum = len(ssp['Teff'])
            
            ii += 1
            print(f'... {ii}: {starnum}')
            # pbar.update(starnum)
            
            if starnum >= nsamp:
                break
        
        # if sampler drew too many samples, trim down to nsamp
        if nsamp > starnum:
            ssp = {kk:jr.choice(rng_key_tmp,ssp[kk],shape=(nsamp,),replace=False) for kk in ssp.keys()}

        if loggboost:

            # At the end of buildssp, after you have combined your samples into ssp:
            # Suppose ssp is a dictionary where ssp['log(g)'] contains the log(g) values.
            # Define bins for log(g) – you can adjust the number of bins as needed.
            logg = ssp['log(g)']
            bins = jnp.linspace(0.0, 5.5, 11)  # 10 bins spanning the log(g) range 0–5.5
            bin_ids = jnp.digitize(logg, bins)
            
            # Count samples per bin:
            counts = jnp.array([jnp.sum(bin_ids == i) for i in range(1, len(bins)+1)])
            
            # Compute weights: samples in low-count bins get higher weight.
            # (Add a small constant to avoid division by zero.)
            weights = 1.0 / (counts[bin_ids - 1] + 1e-6)
            weights = weights / jnp.sum(weights)  # Normalize the weights

            # Use JAX's random.choice to re-sample nsamp indices based on these weights.
            self.rng_key, rng_key_tmp = jr.split(self.rng_key)
            indices = jr.choice(rng_key_tmp, a=jnp.arange(len(logg)), shape=(nsamp,), replace=True, p=weights)
            
            # Build a new, balanced dataset:
            balanced_ssp = {kk: ssp[kk][indices] for kk in ssp.keys()}
            
            return balanced_ssp

        else:
            return ssp


def main(cachedata=True, cachemodel=True, usepredata=False, train=True, restart=True):

    version = '0.8.0'
    restart_version = '0.7.0'

    minmass = 0.25
    maxmass = 5.0
    minage = 0.01
    maxage = 14.0
    
    meanMass = 0.8
    stdMass  = 0.5
    meanlogMass = -0.15
    stdlogMass  = 0.5
    meanAge  = 5.0
    stdAge   = 3.0
    meanTeff = 4850.0
    stdTeff  = 2000.0
    meanlogg = 4.5
    stdlogg  = 1.0

    # # v0.1.0/v0.2.0    
    # nn_depth=4
    # nn_block_dim=32
    # flow_layers=3
    # flowtype = 'BNAF'

    # v0.3.0/v0.4.0/v0.5.0/v0.6.0
    # nn_depth=8
    # nn_block_dim=64
    # flow_layers=3
    # flowtype = 'BNAF'
    
    # v0.7.0
    nn_depth=16
    nn_block_dim=64
    flow_layers=16
    flowtype = 'MAF'

    # # v0.8.0
    # nn_depth=8
    # nn_block_dim=64
    # flow_layers=32
    # flowtype = 'MAF'

    max_patience = 2000
    logg_boost = False
    age_boost = False
    
    mistssp = MIST_dataloader(minmass=minmass,maxmass=maxmass,minage=minage,maxage=maxage)
    
    if train:
        nsamp = 15000
        
        ssp = None
        if usepredata:
            savedsets = np.array(list(glob("data/traindata*.fits")))
            savediter = jnp.array([int(x.split('_')[-1].replace('.fits','')) for x in savedsets])
            cond = savediter >= nsamp
            
            if cond.sum() > 0:
                fname = savedsets[cond]
                ssp_t = Table.read(fname[0],format='fits')
                ssp_t = ssp[:nsamp]
                
        if ssp is None:
            ssp_t = mistssp.buildssp(feh=0.0,afe=0.0,nsamp=nsamp, loggboost=logg_boost, ageboost=age_boost) 
            ssp_t = Table(ssp_t)
        
        ssp = {kk:jnp.array([]) for kk in ssp_t.keys()}
        for kk in ssp_t.keys():
            ssp[kk] = jnp.array(ssp_t[kk],dtype=jnp.float32)

        if cachedata:
            ssp_t.write(f'data/traindata_{nsamp}.fits',overwrite=True)
        
        fig_ssp,axlist_ssp = plt.subplots(nrows=3,ncols=2,layout='constrained')
        axlist_ssp = axlist_ssp.flatten()
        
        axlist_ssp[0].hist(ssp['initial_Mass'],bins=100,range=(0.1,5.0),histtype='step',color='k',lw=1)
        axlist_ssp[1].hist(ssp['Age'],bins=100,range=(0,15),histtype='step',color='k',lw=1)
        axlist_ssp[2].hist(ssp['Teff'],bins=100,range=(3000,7000),histtype='step',color='k',lw=1)
        axlist_ssp[3].hist(ssp['log(g)'],bins=100,range=(0.0,5.5),histtype='step',color='k',lw=1)
        axlist_ssp[4].hist2d(ssp['initial_Mass'],ssp['Age'],bins=(50,50),range=((0.1,5.0),(0,15)),cmap='Greys',norm=colors.LogNorm())
        axlist_ssp[5].hist2d(ssp['Teff'],ssp['log(g)'],bins=(50,50),range=((3000,7000),(0.0,5.5)),cmap='Greys',norm=colors.LogNorm())

        axlist_ssp[0].set_xlabel(r'Mass$_{i}$')
        axlist_ssp[1].set_xlabel('Age [Gyr]')
        axlist_ssp[2].set_xlabel('Teff')
        axlist_ssp[3].set_xlabel('log(g)')

        axlist_ssp[4].set_xlabel('Mass$_{i}$')
        axlist_ssp[4].set_ylabel('Age [Gyr]')
        axlist_ssp[5].set_xlabel('Teff')
        axlist_ssp[5].set_ylabel('log(g)')
        
        axlist_ssp[5].set_ylim(5.5,0.0)
        axlist_ssp[5].set_xlim(7500.0,2500.0)
        
        fig_ssp.savefig(f'trainsample_{version}.png',dpi=200)
        
        print(f'--> Sampling {nsamp} objects:')
        print(f'total mass = {jnp.sum(ssp["Mass"])}, min(mass) = {jnp.min(ssp["Mass"])}, max(mass) = {jnp.max(ssp["Mass"])}')
        print(f'min(Age) = {jnp.min(ssp["Age"])}, max(Age) = {jnp.max(ssp["Age"])}')
        
        x = jnp.array((ssp['Teff'],ssp['log(g)'])).T
        x_s = (x - jnp.array([meanTeff,meanlogg])) / jnp.array([stdTeff,stdlogg])

        u = jnp.array((ssp['initial_Mass'],ssp['Age'])).T
        u_s = (u - jnp.array([meanMass,meanAge])) / jnp.array([stdMass,stdAge])

        key, subkey = jr.split(jr.key(0))

        # Create an exponential decay learning rate schedule.
        lr_schedule = optax.exponential_decay(
            init_value=1e-3,           # initial learning rate
            transition_steps=500,      # decay every 1000 steps (adjust as needed)
            decay_rate=0.99,           # multiply the learning rate by 0.99 each time
            staircase=True
        )

        # Create an AdamW optimizer with weight decay.
        optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)

        if flowtype == 'BNAF':
            flow = block_neural_autoregressive_flow(
                key=subkey,
                base_dist=Normal(jnp.zeros(x_s.shape[1])),
                # activation=bij.Sigmoid(), 
                # activation=bij.LeakyTanh(),
                cond_dim=u_s.shape[1],
                nn_depth=nn_depth,
                nn_block_dim=nn_block_dim,
                flow_layers=flow_layers,
                throw=False,
            )

            if restart:
                if os.path.isfile(f"./model/block_neural_{restart_version}.eqx"):
                    flow = eqx.tree_deserialise_leaves(f"./model/block_neural_{restart_version}.eqx",flow)

        if flowtype == "MAF":
            flow = masked_autoregressive_flow(
                key=subkey,
                base_dist=Normal(jnp.zeros(x_s.shape[1])),
                cond_dim=u_s.shape[1],
                flow_layers=flow_layers,
                nn_depth=nn_depth,
                nn_width=nn_block_dim,
                # nn_activation=bij.Sigmoid(), 
                # throw=False,
            )
            if restart:
                if os.path.isfile(f"./model/masked_autoregressive_{restart_version}.eqx"):
                    flow = eqx.tree_deserialise_leaves(f"./model/masked_autoregressive_{restart_version}.eqx",flow)

                
        flow, losses = fit_to_data(
            key=subkey,
            dist=flow,
            x=x_s,
            optimizer=optimizer, 
            # learning_rate=5e-4, 
            max_patience=max_patience,
            max_epochs=50000,
            batch_size=3000,
            condition=u_s,
        )
        
        # plot losses and validation
        fig,ax = plt.subplots(nrows=1,ncols=1,layout='constrained')
        ax.plot(losses['train'],label='train',lw=0.5,alpha=0.5)
        ax.plot(losses['val'],label='val',lw=0.5,alpha=0.5)
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_ylim(scoreatpercentile(losses['train'][50:],1)-0.5,scoreatpercentile(losses['val'][50:],99.9)+1.0)
        fig.savefig(f'loss_{version}.png',dpi=200)
        
        if cachemodel:
            if flowtype == "BNAF":
                eqx.tree_serialise_leaves(f"./model/block_neural_{version}.eqx",flow)
            if flowtype == "MAF":
                eqx.tree_serialise_leaves(f"./model/masked_autoregressive_{version}.eqx",flow)
    else:
        key, subkey = jr.split(jr.key(0))
        if flowtype == 'BNAF':
            flow = block_neural_autoregressive_flow(
                key=subkey,
                base_dist=Normal(jnp.zeros(2)),
                # activation=bij.Sigmoid(), 
                # activation=bij.LeakyTanh(),
                cond_dim=2,
                nn_depth=nn_depth,
                nn_block_dim=nn_block_dim,
                flow_layers=flow_layers,
                # throw=False,
            )
            flow = eqx.tree_deserialise_leaves(f"./model/block_neural_{version}.eqx",flow)
        if flowtype == "MAF":
            flow = masked_autoregressive_flow(
                key=subkey,
                base_dist=Normal(jnp.zeros(2)),
                cond_dim=2,
                flow_layers=flow_layers,
                nn_depth=nn_depth,
                nn_width=nn_block_dim,
                # nn_activation=bij.Sigmoid(), 
                # throw=False,
            )
            flow = eqx.tree_deserialise_leaves(f"./model/masked_autoregressive_{version}.eqx",flow)
        x = ''
        u = ''
        ssp = ''
        
    fig,axlist = plt.subplots(nrows=2,ncols=2,layout='constrained')
    axlist = axlist.flatten()

    testages = [2.0,4.0,6.0,8.0]

    for ii,ax in enumerate(axlist):

        ax.text(0.99,0.9,f'{testages[ii]} Gyr',transform=ax.transAxes,ha='right',va='center')

        # resolution = 10
        # # massarr = jnp.linspace(0.25,1.5,1000)
        # massarr = jnp.ones(resolution*resolution)
        # massarr = (massarr - u.mean(axis=0)) / u.std(axis=0)
        # massarr = jnp.array([massarr]).T

        # xgrid, ygrid = jnp.meshgrid(
        #     jnp.linspace(3000, 7000, resolution), jnp.linspace(1.0, 5.5, resolution),
        # )
        # xyinput = jnp.array((xgrid.flatten(), ygrid.flatten())).T
        # xyinput = (xyinput - x.mean(axis=0)) / x.std(axis=0)

        # lnprob = flow.log_prob(xyinput, massarr)
        # zgrid = lnprob.reshape(resolution, resolution)
        # zgrid = jnp.exp(lnprob)
        # ax.contour(xgrid,ygrid,zgrid,levels=50)

        print(f'--> Sampling flow for Prob Map')
        nsamp = 10000
        nbatch = 1

        if os.path.isfile(f"./data/testdata_{testages[ii]}gyr.fits"):
            ssp_iso = Table.read(f"./data/testdata_{testages[ii]}gyr.fits")
            if len(ssp_iso) < nsamp:
                ssp_iso = mistssp.buildssp(age=testages[ii],feh=0.0,afe=0.0,nsamp=nsamp)
                ssp_iso = Table(ssp_iso)
                ssp_iso.write(f"./data/testdata_{testages[ii]}gyr.fits",overwrite=True)
            else:
                ssp_iso = ssp_iso[:nsamp]
        else:
            ssp_iso = mistssp.buildssp(age=testages[ii],feh=0.0,afe=0.0,nsamp=nsamp)
            ssp_iso = Table(ssp_iso)
            ssp_iso.write(f"./data/testdata_{testages[ii]}gyr.fits",overwrite=True)
        
        massarr = jnp.linspace(ssp_iso['initial_Mass'].min(),ssp_iso['initial_Mass'].max(),nsamp)
        agearr = testages[ii] * jnp.ones(nsamp)

        u_t = jnp.array((massarr,agearr)).T
        u_t = (u_t - jnp.array([meanMass,meanAge])) / jnp.array([stdMass,stdAge])
            
        samp_s = flow.sample(subkey, (nbatch,), condition=u_t)
        samp = samp_s * jnp.array([stdTeff,stdlogg]) + jnp.array([meanTeff,meanlogg])
        samp = samp.reshape(-1,2)

        ax.scatter(ssp_iso['Teff'],ssp_iso['log(g)'],marker='.',s=20,c='k',alpha=0.1,ec='none')
                
        ax.scatter(samp[:,0],samp[:,1],marker='.',c='C3',s=10,alpha=0.25,ec='none')
        # ax.hist2d(samp[:,0],samp[:,1],bins=(250,250),range=((3000,7000),(1.5,5.5)),cmap='Greys',
        #           norm=colors.LogNorm(),alpha=0.5)

        print(f'--> Sampling flow using IMF')
        nbatch = 1
        
        imf = IMF_Prior(low=ssp_iso['initial_Mass'].min(),high=ssp_iso['initial_Mass'].max())
        
        massarr = imf.sample(n=nsamp)
        agearr = testages[ii] * jnp.ones(nsamp)
        u_t = jnp.array((massarr,agearr)).T
        u_t = (u_t - jnp.array([meanMass,meanAge])) / jnp.array([stdMass,stdAge])
            
        samp_s = flow.sample(subkey, (nbatch,), condition=u_t)
        samp = samp_s * jnp.array([stdTeff,stdlogg]) + jnp.array([meanTeff,meanlogg])
        samp = samp.reshape(-1,2)

        ax.scatter(samp[:,0],samp[:,1],marker='.',c='C0',s=10,alpha=0.5,ec='none')

        ax.set_xlim(7500.0,2500.0)
        ax.set_ylim(5.5,1.0)
        ax.set_xlabel('Teff')
        ax.set_ylabel('log(g)')


    fig.savefig(f'test_{version}.png',dpi=500)
        
    return flow,x,u,ssp
                    
if __name__ == '__main__':
    flow,x,u,ssp = main()