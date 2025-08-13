import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import kcorrections as k
import numpy as np
from scipy import stats
from astropy.table import join,Table,Column,vstack
from kcorrections  import DESI_KCorrection 
from rootfinders import root_itp,root_sec,root_itp2
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import gaussian_filter
from desiutil.plots import prepare_data, init_sky, plot_grid_map, plot_healpix_map, plot_sky_circles, plot_sky_binned
cosmo = FlatLambdaCDM(H0=100, Om0=0.313, Tcmb0=2.725)   #Standard Planck Cosmology in Mpc/h units



########################################################################################################################    
    
#Define Sample selection cuts and their display styles
#and other "global" variables so they can be defined in any routine
def selection(reg):
    f_ran=1.0   #Random down smapling factor
    Qevol=0.78 #0.78 #Assumed luminosity evolution parameter
    area_N_Y1=2393.4228/(4*np.pi*(180.0/np.pi)**2)
    area_S_Y1=5358.2728/(4*np.pi*(180.0/np.pi)**2)
    area_N_Y3=3827.50/(4*np.pi*(180.0/np.pi)**2) #from assuming 2500 randoms/sqdeg in catalog 0 
    area_S_Y3=8527.58/(4*np.pi*(180.0/np.pi)**2)
    South={'zmin': 0.002, 'zmax': 0.6, 'bright': 10.0, 'faint': 19.5 ,\
          'area': area_S_Y3, 'col': 'red' , 'style': 'solid', 'f_ran': f_ran, 'Qevol': Qevol}
    North={'zmin': 0.002, 'zmax': 0.6, 'bright': 10.0, 'faint': 19.54,\
          'area': area_N_Y3, 'col': 'blue', 'style': 'dashed', 'f_ran': f_ran, 'Qevol': Qevol}
    if (reg=='N'):
        x=North
    elif (reg=='S'):
        x=South
    else:
        print('Selection(region): unknown region')  
    return x
########################################################################################################################
# load in catalogues
def Y3load_catalogues(fpath):
    dat = Table.read(fpath)
    # Copy of PHOTSYS which is N/S to the "reg" column this code uses
    dat.add_column(Column(name='reg', data=dat['PHOTSYS']))

    # Place holders for addtional quantitites that will be calculated 
    dat.add_column(Column(name='REST_GMR_0P1', data=np.zeros(dat['Z'].size)))
    dat.add_column(Column(name='ABSMAG_RP1', data=np.zeros(dat['Z'].size)))
    dat.add_column(Column(name='ijack', data=np.zeros(dat['Z'].size, dtype=int)))
    

    #Apparent magnitudes avoiding infinities where fluxes are 0 or negative
    dat.add_column(Column(name='gmag', data=22.5-2.5*np.log10(np.clip(dat['flux_g_dered'],1.0e-10,None)) ))
    dat.add_column(Column(name='rmag', data=22.5-2.5*np.log10(dat['flux_r_dered']) ))
    dat.add_column(Column(name='zmag', data=22.5-2.5*np.log10(np.clip(dat['flux_z_dered'],1.0e-10,None)) ))

    #Add Observerframe colour 
    dat.add_column(Column(name='gmr_obs', data=dat['gmag']-dat['rmag']))


    return dat
########################################################################################################################
# load in catalogues
def load_catalogues(fpathN,fpathS):
    datS = Table.read(fpathS)
    datS.add_column(Column(name='reg', data=["S" for x in range(datS['Z'].size)]))
    datN = Table.read(fpathN)
    datN.add_column(Column(name='reg', data=["N" for x in range(datN['Z'].size)]))

    #Combine into a single table with the two parts flagged by the value in 
    #the reg column and delete the original tables
    dat=vstack(([datS, datN]), join_type='exact', metadata_conflicts='warn')
    del datS,datN

    #remove z_ref=0.0 columns to avoid mis-use
    dat.remove_column('ABSMAG_RP0')
    dat.remove_column('REST_GMR_0P0')
    dat['REST_GMR_0P1']=0.0


    # Add derived columns

    #Apparent magnitudes
    dat.add_column(Column(name='gmag', data=22.5-2.5*np.log10(dat['flux_g_dered']) ))
    dat.add_column(Column(name='rmag', data=22.5-2.5*np.log10(dat['flux_r_dered']) ))
    dat.add_column(Column(name='zmag',\
                data=22.5-2.5*np.log10(np.clip(dat['flux_z_dered'],1.0e-10,None)) ))

    #Observerframe colour
    dat.add_column(Column(name='gmr_obs', data=dat['gmag']-dat['rmag']))
#    dat.add_column(Column(name='OBS_GMR', data=dat['gmag']-dat['rmag']))

    return dat
########################################################################################################################
# load in catalogue
def load_catalogue(fpath):
    dat = Table.read(fpath)
    dat.add_column(Column(name='reg', data=dat['PHOTSYS'] ))
    # dummy arrays in which to later store the correct quantities
    dat.add_column(Column(name='REST_GMR_0P1', data=dat['flux_g_dered'] ))
    dat.add_column(Column(name='REST_GMR_0P0', data=dat['flux_g_dered'] ))
    dat.add_column(Column(name='ABSMAG_RP1', data=dat['flux_g_dered'] ))
    dat.add_column(Column(name='ABSMAG_RP0', data=dat['flux_g_dered'] ))


    
    dat.info('stats')
    # Add derived columns

    #Apparent magnitudes
    dat.add_column(Column(name='gmag',\
                data=22.5-2.5*np.log10(np.clip(dat['flux_g_dered'],1.0e-10,None)) ))
    dat.add_column(Column(name='rmag',\
                data=22.5-2.5*np.log10(np.clip(dat['flux_r_dered'],1.0e-10,None)) ))
    dat.add_column(Column(name='zmag',\
                data=22.5-2.5*np.log10(np.clip(dat['flux_z_dered'],1.0e-10,None)) ))

    #Observerframe colour
    dat.add_column(Column(name='gmr_obs', data=np.clip((dat['gmag']-dat['rmag']),None,3.76)))
    dat.add_column(Column(name='OBS_GMR', data=np.clip((dat['gmag']-dat['rmag']),None,3.76)))

    return dat

########################################################################################################################

def solve_jackknife_nonsq(data, ndiv_ra=4, ndiv_dec=5, offset=275):
    '''
    Find the boundaries to split up data into jackknife areas based on (ra, dec) in (ndiv_ra x ndiv_dec) chunks.
    The offset allows the RA edge of the first bin to be chosen.
    '''           
    print('length of data table:',len(data))        
    njack         = ndiv_ra * ndiv_dec
    #jk_volfrac    = (njack - 1.) / njack 

    dpercentile_ra   = 100. / ndiv_ra
    dpercentile_dec  = 100. / ndiv_dec

    percentiles_ra   = np.arange(dpercentile_ra, 100. + dpercentile_ra, dpercentile_ra)
    percentiles_dec   = np.arange(dpercentile_dec, 100. + dpercentile_dec, dpercentile_dec)
    #print(percentiles_ra)
    #print(percentiles_dec)
    
    
    # Set up a astropy table to store the RA and Dec values of the boundaries initialised to zero
    limits        = Table()
    limits['ralow'] = np.zeros(njack)
    limits['rahigh'] = np.zeros(njack)
    limits['declow'] = np.zeros(njack)
    limits['dechigh'] = np.zeros(njack)

    jk = 0
    # create a temporary version of the RA angle shifted by the selected offset modulo 360 degrees
    RA_JK = (data['RA'] + offset) % 360

    for ra_per in percentiles_ra:
        #Find the next RA band
        # Given a vector V of length N, the q-th percentile of V is the q-th ranked value in a sorted copy of V. 
        # https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.percentile.html
        rahigh    = np.percentile(RA_JK, ra_per)
        ralow     = np.percentile(RA_JK, ra_per - dpercentile_ra)

        #print('processing RA band:')
        #print('{:.6f}\t{:.6f}'.format(ralow, rahigh))

        for dec_per in percentiles_dec:
            #define mask to select data in this RA band
            isin    = (RA_JK >= ralow) & (RA_JK <= rahigh)

            # find the next dec band
            dechigh = np.percentile(data[f'DEC'][isin], dec_per)
            declow  = np.percentile(data[f'DEC'][isin], dec_per - dpercentile_dec)

            #print('processing dec band:')
            #print('\t{:.6f}\t{:.6f}'.format(declow, dechigh))

            #store the limits of this band in the limits table 
            limits['ralow'][jk]   = ralow
            limits['rahigh'][jk]  = rahigh
            limits['declow'][jk]  = declow
            limits['dechigh'][jk] = dechigh
            
            # move onto the next region
            jk     += 1

    # before returning remove the offset applied to the RA values
    # create a version of the RA angle shifted by the selected offset modulo 360 degrees
    limits['ralow'] = (limits['ralow'] - offset) % 360
    limits['rahigh'] = (limits['rahigh'] - offset) % 360
    del RA_JK

    return  njack, limits
###########################################################################################################################
#  Set jackknife indices for the objects in data[regmask]
def set_jackknife(dat, regmask, limits, noffset, njack):
  
  for ijack in range(0, njack ):  
         # mask to select data in the ijack'th jackknife region
         if (limits['ralow'][ijack]<limits['rahigh'][ijack]): #if window does not straddle the 0/360 deg boundary as a result of applying the offset then the mask is simple
            injack = regmask & (dat['RA']>= limits['ralow'][ijack]) & (dat['RA']<= limits['rahigh'][ijack]) & (dat['DEC']>= limits['declow'][ijack]) & (dat['DEC']<= limits['dechigh'][ijack])
         else:  # otherwise we have to use this more complicated version
              injack = regmask & (((dat['RA']>= 0.0 ) & (dat['RA']<= limits['rahigh'][ijack])) | ((dat['RA']>=  limits['ralow'][ijack]) & (dat['RA']<=360.0))) & (dat['DEC']>= limits['declow'][ijack]) & (dat['DEC']<= limits['dechigh'][ijack])
         print('assigning',np.count_nonzero(injack),'objects to jackknife region',ijack+noffset)
         dat['ijack'][injack]=ijack+noffset

  #   The above ought to assign all objects to a jackknife region but rounding errors mean some escape. 
  #  The following code assigns any escapees to the nearest jackknife region by expanding each region by an amount eps.  
  if ( np.count_nonzero(dat['ijack'][regmask]==-999) ):
        print('Attempting to assign the remaining unassigned  ',np.count_nonzero(dat['ijack'][regmask]==-999),' objects in this region')
        eps=1.0 #1 degree!!
        mask= regmask & (dat['ijack']==-999) # just the objects in this region that have not been assigned
        for ijack in range(0, njack ):
          # mask to select data in the ijack'th jackknife region
          if (limits['ralow'][ijack]<limits['rahigh'][ijack]): #if window does not straddle the 0/360 deg boundary as a result of applying the offset then the mask is simple
            injack = mask & (dat['RA']>= limits['ralow'][ijack]-eps) & (dat['RA']<= limits['rahigh'][ijack]+eps) & (dat['DEC']>= limits['declow'][ijack]-eps) & (dat['DEC']<= limits['dechigh'][ijack]+eps)
          else:  # otherwise we have to use this more complicated version
            injack = mask & (((dat['RA']>= 0.0 ) & (dat['RA']<= limits['rahigh'][ijack]+eps)) | ((dat['RA']>=  limits['ralow'][ijack]-eps) & (dat['RA']<=360.0))) & (dat['DEC']>= limits['declow'][ijack]-eps) & (dat['DEC']<= limits['dechigh'][ijack]+eps)
          if(np.count_nonzero(injack)>0):
            print('reassigning',np.count_nonzero(injack),'an additional objects to jackknife region',ijack+noffset)
            dat['ijack'][injack]=ijack+noffset
  # report if still some unassigned            
  if ( np.count_nonzero(dat['ijack'][regmask]==-999) ):
        print('There are still ',np.count_nonzero(dat['ijack'][regmask]==-999),'unassigned objects in this region')          
  return

##########################################################################################################################
def z_tozLG(dat):
    """Transform redshift from Heliocentric to Local Group velocity frame.
    Tested on data in GAMA catalogue https://www.gama-survey.org/dr4/schema/table.php?id=488
    """
    
    #Velocity of the sun baryocentre in the Local Group Frame
    v_LG=316.0 # km/s  NED and AJ 111, 794, 1996
    RA_LG=np.deg2rad(325.371)  # l=93deg b=-4deg also from NED converted to RA,DEC by NED and to radians here
    DEC_LG=np.deg2rad(47.527)  
    #Projection of this velocity along object RA and dec. Note objects RA and dec in degrees but LG already converted to radians
    v_proj=v_LG*(np.sin(np.deg2rad(dat['DEC']))*np.sin(DEC_LG)+np.cos(np.deg2rad(dat['DEC']))*np.cos(DEC_LG)*np.cos(np.deg2rad(dat['RA'])-RA_LG))
    #Compute redshift in the LG frame assuming input redshift is Heliocentric
    dat.add_column(Column(name='ZLG', data= (1.0+dat['Z'])*(1.0+v_proj/(299792.458))-1.0 ))
    return
########################################################################################################################
# ABSMAG_R= appmag -DMOD  -kcorr_r.k(z, rest_GMR) +Qevol*(z-0.1) 
def ABSMAG(appmag,z,rest_GMR,kcorr_r,Qevol):
        DMOD=25.0+5.0*np.log10(cosmo.luminosity_distance(z).value)  
        ABSMAG=appmag-DMOD-kcorr_r.k(z,rest_GMR)+Qevol*(z-0.1)
        return ABSMAG
    
# Make plots of the k-corrections to check they are sensible and smooth in both redshift and colour
########################################################################################################################

def recompute_rest_col_mag(dat,regions, fsf, fresh=False, plot=True):
    
    for reg in regions:
      print('Computing restframe colours for region ',reg)
      Sel=selection(reg) # define selection parameters for this region
      regmask=(dat['reg']==reg)#mask to select objects in specified region

      if fresh == True:
          fsfmask=(fsf['PHOTSYS']==reg)
          k.construct_colour_table(fsf, reg, plot=plot)
      else: 
          print('using precomputed k-corrections')         
          
      # call to use the lookup table to assign rest-frame colours                       
      k.colour_table_lookup(dat, regmask, reg, replot=plot, fresh=False) 

      # call to assign k-corrected magnitudes  
      kcorr_rM  = DESI_KCorrection(band='R', file='jmext', photsys=reg) #set k-correction for region
      dat['ABSMAG_RP1'][regmask]=ABSMAG(dat['rmag'][regmask],dat['Z'][regmask],dat['REST_GMR_0P1'][regmask],kcorr_rM,Sel['Qevol'])


    return


########################################################################################################################    

#Compute zmax,zmin and along with v and vmax add to the data table    
def compute_zmax_vmax(dat,regions):


    # We want to solve
    # appmag=ABSMAG_R+ DMOD  +kcorr_r.k(z, rest_GMR) -Qevol*(z-0.1) = maglimit [19.5 for BSG S]
    # which is equivalent to 
    # 'magdiff' = maglimit-ABSMAG_R = DMOD  +kcorr_r.k(z, rest_GMR) -Qevol*(z-0.1) 
    # Difference in apparent and absolute magnitude, magapp-ABSMAG, as a function of redshift
    def magdiff(z,rest_GMR,Qevol):
        DMOD=25.0+5.0*np.log10(np.maximum(cosmo.luminosity_distance(z).value,1.0e-08)) 
        magdiff =  DMOD  +kcorr_r.k(z, rest_GMR) -Qevol*(z-0.1)
        return magdiff

    vmax=np.zeros(dat['Z'].size) # set up array ready to receive vmax values
    vmin=np.zeros(dat['Z'].size) # set up array ready to receive vmin values
    v=np.zeros(dat['Z'].size) # v values
    zmax=np.zeros(dat['Z'].size) # and zmax values
    zmin=np.zeros(dat['Z'].size) # and zmin values
    for reg in regions:
        print('starting region ',reg)
        Sel=selection(reg) # define selection parameters for this region
        #set up k-corrections for region
        kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg) #set k-correction for region
        regmask=(dat['reg']==reg)#mask to select objects in specified region
        mask = regmask & (dat['rmag']<Sel['faint']) # in region and brighter than faint cut
        magdiff_target=Sel['faint']-dat['ABSMAG_RP1']
        #The zmax we want has to be between the following two values
        zguess0=np.copy(dat['Z'])
        zguess1=Sel['zmax']+0.0*zguess0 # "sample_zmax"
        #If the root is at greater than zguess1="sample_zmax" the root finder will just return zguess1  (the b limit in the rootfinder code)
        ztol=0.00001
        zmax[mask]=root_itp2(magdiff,dat['REST_GMR_0P1'][mask],Sel['Qevol'],magdiff_target[mask],ztol,zguess0[mask],zguess1[mask])
        print('zmax values found')

        magdiff_target=Sel['bright']-dat['ABSMAG_RP1']
        mask = regmask & (dat['rmag']>Sel['bright']) # in region and fainter than bright cut
        #The zmin we want has to be between the following two values
        zguess0=np.zeros(dat['Z'].size)+Sel['zmin'] # "sample_zmin"
        zguess1=np.copy(dat['Z'])
        #They are fed into the root finder in reverse order so that if root is at less than zguess0="sample_zmin" the root finder
        # will just return zguess0 (the b limit in the root findr code) which is what we want.
        ztol=0.00001
        zmin[mask]=root_itp2(magdiff,dat['REST_GMR_0P1'][mask],Sel['Qevol'],magdiff_target[mask],ztol,zguess1[mask],zguess0[mask]) 
        print('zmin values found')
        zcheck=  (zmin[mask]-Sel['zmin'])
        if (np.min(zcheck)<0): print('ERROR: object with zmin<Sel[zmin]')

        #Apply selection bounds and compute v and vmax
        print('applying selection limits and computing vmax (vmax-vmin) and v (v-vmin) but also vmin')
        if ((zmax[regmask].max()>Sel['zmax']).any()): print('Error: Some zmax values greater than sample zmax cut')
        if ((zmin[regmask].min()<Sel['zmin']).any()): print('Error: Some zmin values less than sample zmin cut')
        v[regmask]=Sel['area']*(4.0*np.pi/3.0)*\
        ( (cosmo.comoving_distance(dat['Z'][regmask]).value)**3 -(cosmo.comoving_distance(zmin[regmask]).value)**3 )
        
        vmax[regmask]=Sel['area']*(4.0*np.pi/3.0)*\
        ( (cosmo.comoving_distance(zmax[regmask]).value)**3 -(cosmo.comoving_distance(zmin[regmask]).value)**3 )
        vmin[regmask]=Sel['area']*(4.0*np.pi/3.0)*(cosmo.comoving_distance(zmin[regmask]).value)**3 
   
    dat.add_column(Column(name='zmax', data=zmax))
    dat.add_column(Column(name='zmin', data=zmin))
    dat.add_column(Column(name='v', data=v))
    dat.add_column(Column(name='vmax', data=vmax))
    dat.add_column(Column(name='vmin', data=vmin))
    del zguess0,zguess1,zmin,zmax,v,vmax,vmin,mask,regmask   #tidy up  
    return
####################################################
def plot_kcorr(regions):
    # extract the default colour sequence to have more control of line colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for reg in regions:
        # set up the k-corrections for this photometry system region
        kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg)
        Sel=selection(reg) # define selection parameters for this region
        z=np.linspace(0.0,0.6,300)
        icol=-1
        #for rest_GMR in np.array([0.39861393, 0.53434181, 0.6534462 , 0.76661587, 0.86391068,
       #0.93073082, 0.9832058 ]): #
            
        for rest_GMR in np.linspace(0.0,1.1,8):   
            GMR=rest_GMR*np.ones(z.size)
            icol += 1
            k=kcorr_r.k(z, GMR)
            label=reg+': G-R='+np.array2string(rest_GMR)
            plt.plot(z,k,label=label,color=colors[icol],linestyle=Sel['style']) 
    plt.xlabel('$z$')    
    plt.ylabel('$k^r(z)$')  
    plt.legend(loc=(1.04,0))    
    plt.show()


    return

########################################################################################################################

#Plot how z_max depends on absolute magnitude and colour code by rest frame colour
def plot_zmax_absmag(dat):
    cmap= plt.get_cmap('jet')
    col=np.clip(((dat['REST_GMR_0P1']+0.5)/2.0),0.0,1.0)
    plt.scatter(dat['ABSMAG_RP1'],dat['zmax'], marker='.', c=col ,cmap=cmap, linewidths=0,s=0.25,alpha=0.2,label='colour coded by rest frame colour')
    plt.xlim([-12,-23])
    plt.ylim([0.0,0.6])
    plt.ylabel('$z_{max}$')
    plt.xlabel('$M_r - 5 \log h$')
    plt.legend()
    plt.show()
    return
########################################################################################################################
#Plot how z_min depends on absolute magnitude and colour code by rest frame colour
def plot_zmin_absmag(dat):
    cmap= plt.get_cmap('jet')
    col=np.clip(((dat['REST_GMR_0P1']+0.5)/2.0),0.0,1.0)
    plt.scatter(dat['ABSMAG_RP1'],dat['zmin'], marker='.', c=col ,cmap=cmap, linewidths=0,s=0.25,alpha=0.2,label='colour coded be rest-frame colour')
    plt.xlim([-12,-23])
    #plt.ylim([0.0,0.1])
    plt.ylabel('$z_{min}$')
    plt.xlabel('$M_r - 5 \log h$')
    plt.legend()
    plt.show()
    return

########################################################################################################################

#Plot how z_max depends z colour code by absolute magnitude
def plot_zmax_z(dat):
    cmap= plt.get_cmap('jet')
    col=np.clip(((dat['ABSMAG_RP1']+22)/10.0),0.0,1.0)
    plt.scatter(dat['Z'],dat['zmax'], marker='.', c=col ,cmap=cmap, linewidths=0,s=0.25,alpha=0.2,label='no point should have z>zmax')
    plt.xlim([0.0,0.6])
    plt.ylim([0.0,0.6])
    plt.ylabel('$z_{max}$')
    plt.xlabel('$z$')
    plt.legend()
    plt.show()
    return

########################################################################################################################

#Plot how z_min depends z colour code by absolute magnitude
def plot_zmin_z(dat):
    cmap= plt.get_cmap('jet')
    col=np.clip(((dat['ABSMAG_RP1']+22)/10.0),0.0,1.0)
    plt.scatter(dat['Z'],dat['zmin'], marker='.', c=col ,cmap=cmap, linewidths=0,s=0.25,alpha=0.2,label='no point should have z<zmin')
    plt.xlim([0.0,0.6])
    #plt.ylim([0.0,0.6])
    plt.ylabel('$z_{min}$')
    plt.xlabel('$z$')
    plt.legend()
    plt.show()
    return



########################################################################################################################
 ##################################################################
# make a histogram comparing the redshift distributions to selection limits
def hist_nz(dat,ran,regions):
    bins = np.arange(0, 0.6, 0.01)
    bin_cen=(bins[:-1] + bins[1:]) / 2.0
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg) #set k-correction for region
        regmask=(dat['reg']==reg)#mask to select objects in specified region    
        mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['rmag'] < Sel['faint']) & (dat['Z'] < Sel['zmax']) & (dat['rmag'] > Sel['bright']) #sample selection
        wcount,binz=np.histogram(dat[mask]['Z'], bins=bins,  density=True, weights=dat[mask]['WEIGHT'])
        count,binz=np.histogram(dat[mask]['Z'], bins=bins,  density=True)
        plt.plot(bin_cen, wcount, label='Weighted '+reg, color=Sel['col'], linestyle=Sel['style'],linewidth=2.0)
        plt.plot(bin_cen, count, label=reg, linewidth=1.0, color=Sel['col'], linestyle=Sel['style'])  
    #Repeat for the randoms             
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg) #set k-correction for region
        regmask=(ran['reg']==reg)#mask to select objects in specified region    
        mask = (regmask) & (ran['Z'] > Sel['zmin']) & (ran['rmag'] < Sel['faint']) & (ran['Z'] < Sel['zmax']) & (ran['rmag'] > Sel['bright']) #sample selection
        wcount,binz=np.histogram(ran[mask]['Z'], bins=bins,  density=True, weights=ran[mask]['WEIGHT'])
        count,binz=np.histogram(ran[mask]['Z'], bins=bins,  density=True)
        plt.plot(bin_cen, wcount, label='Weighted '+reg, color=Sel['col'], linestyle=Sel['style'], linewidth=0.5)
        plt.plot(bin_cen, count, label='randoms'+reg, color=Sel['col'], linestyle=Sel['style'], linewidth=0.5)
    plt.xlabel('z')
    plt.ylabel('dP/dz')
    plt.xlim([0,0.6])
    plt.ylim([0,4.4])
    plt.legend()
    plt.show()
    return
#################################################################################################################
# plot the V/Vmax distribution for the selected sample
def plot_v_vmax(dat,regions):
    bin_edges = np.linspace(0.0, 1.0, 50)
    dbin=bin_edges[1]-bin_edges[0]
    #v_vmax=np.copy(dat['vmax'])
    
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        regmask=(dat['reg']==reg)#mask to select objects in specified region    
        mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['Z'] < Sel['zmax']) & (dat['rmag'] > Sel['bright']) & (dat['rmag'] < Sel['faint']) #sample selection
        plt.hist(dat['v'][mask]/dat['vmax'][mask], bins=bin_edges, histtype='step', density=True, weights=dat[mask]['WEIGHT'],  label='Weighted '+reg)
        plt.hist(dat['v'][mask]/dat['vmax'][mask], bins=bin_edges, histtype='step', density=True, label=reg )
    plt.xlabel('$V/V_{max}$')
    plt.ylabel('$P(V/V_{max})$')
    plt.plot([0.0,1.0],[1.0,1.0],color='black')
    plt.xlim([0,1.0])
    plt.ylim([0.8,1.2])
    plt.legend()
    plt.show()
    return
######################################################################################################################
# magnitude-reshifts scatterplot
# to check data extends to the selection limits
def plot_mag_z(dat,regions,contours=False):
    # Plotting ranges
    range=[[0.0,0.65],[15.5,19.55]]
    sigma=1.0 # smoothing to apply before making contours

    def flatten(l):        # limits of data to use in histogram()
        return [item for sublist in l for item in sublist]

    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        regmask=(dat['reg']==reg)#mask to select objects in specified region  
        # Selection to remove stars and apply faint magnitude
        mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['Z'] < Sel['zmax']) &  (dat['rmag'] < Sel['faint']) #sample selection
        print('All:',reg,'z range:',dat[regmask]['Z'].min(),dat[regmask]['Z'].max())
        print('Selected:',reg,'z range:',dat[regmask]['Z'].min(),dat[regmask]['Z'].max())
        print('All:',reg,'rmag range:',dat[regmask]['rmag'].min(),dat[regmask]['rmag'].max())
        print('Selected:',reg,'rmag range:',dat[regmask]['rmag'].min(),dat[regmask]['rmag'].max())
        # scatter plot
        plt.scatter(dat[mask]['Z'], dat[mask]['rmag'], marker='.', linewidths=0, alpha=0.5, s=0.25, color=Sel['col'])
        # plot limits
        plt.plot(range[0],[Sel['faint'],Sel['faint']], color=Sel['col'],linestyle='solid',linewidth=0.2)
        plt.plot(range[0],[Sel['bright'],Sel['bright']], color=Sel['col'],linestyle='solid',linewidth=0.2)
        plt.plot([Sel['zmin'],Sel['zmin']],range[1], color=Sel['col'],linestyle='solid',linewidth=0.2)
        plt.plot([Sel['zmax'],Sel['zmax']],range[1], color=Sel['col'],linestyle='solid',linewidth=0.2)
        if contours:
            extent=flatten(range)  # flattened version to use in contour()
            counts,x,y=np.histogram2d(dat[mask]['Z'], dat[mask]['rmag'],bins=100, range=range)
            counts=gaussian_filter(counts,sigma)*1.0e+06/np.count_nonzero(mask) #contour units arbitrary!
            plt.contour(counts.transpose(),extent=extent,levels=[20,40,80,160,320,640],colors=Sel['col'])
    

    plt.xlim(range[0])
    plt.ylim(range[1])
    plt.xlabel('z')
    plt.ylabel('r')
    plt.show()
    return
######################################################################################################################
# Colour-Magnitude Scatter plot with marginal histograms
def plot_col_mag(dat,regions):
    def flatten(l):        # limits of data to use in histogram()
        return [item for sublist in l for item in sublist]

    range=[[-0.1,1.3],[-23.99,-14]] # range to use for colour and absolute magnitude limits (both selection and plotting limits)
    nbin=100                       # number of bins per dimension
    sigma=1                        # gaussian smoothing in units of bin width
    levels= [160,320,640,1280,2560,5120,10240,20480] # contour levels in points per bin

    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.0, hspace=0.0)
    # Create the Axes.
    axcolmag = fig.add_subplot(gs[1, 0])
    axcolmag_histx = fig.add_subplot(gs[0, 0], sharex=axcolmag)
    axcolmag_histy = fig.add_subplot(gs[1, 1], sharey=axcolmag)

    axcolmag_histx.tick_params(axis="x", labelbottom=False) #suppress default axis labels
    axcolmag_histy.tick_params(axis="y", labelleft=False)

    

    #Marginal Colour distribution
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask)
        hist,x=np.histogram(dat['REST_GMR_0P1'][mask],bins=nbin,range=range[0])
        hist=hist/Sel['area']
        bin_centres= 0.5*(x[1:]+x[:-1])
        axcolmag_histx.plot(bin_centres,hist,color=Sel['col'])
        ymax=1.1*hist.max()
        axcolmag_histx.set(ylim=[0,ymax])

    #Marginal absolute magnitude distribution
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask)
        hist,x=np.histogram(dat['ABSMAG_RP1'][mask],bins=nbin,range=range[1])
        hist=hist/Sel['area']
        bin_centres= 0.5*(x[1:]+x[:-1])
        axcolmag_histy.plot(hist,bin_centres,color=Sel['col'])
        xmax=1.1*hist.max()
        axcolmag_histy.set(xlim=[0,xmax])
        
    #2D scatter plots
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask)
        axcolmag.scatter(dat[mask]['REST_GMR_0P1'],dat[mask]['ABSMAG_RP1'], marker='.', linewidths=0,s=0.25,color=Sel['col'],alpha=0.1,label='South')
        #Contours of the point distribution in the scatter plots
        extent=flatten(range)  # flattened version to use in contour()
        counts,x,y=np.histogram2d(dat['REST_GMR_0P1'][mask],dat['ABSMAG_RP1'][mask],bins=nbin,range=range)
        counts=gaussian_filter(counts,sigma)/Sel['area']
        axcolmag.contour(counts.transpose(),extent=extent,levels=levels, colors=Sel['col'])



    range[1].reverse()  #flip the Absolute magnitude axes direction
    axcolmag.set(xlabel='$M_g - M_r$', ylabel='$M_r -5 \log h$',xlim=range[0],ylim=range[1])
    Sel=selection('N')
    colN=Sel['col']
    Sel=selection('S')
    colS=Sel['col']
    axcolmag.legend(['North','South'],loc='upper left',labelcolor=[colN,colS])
    
    
    del mask,smask,regmask # tidy up
    del axcolmag,axcolmag_histx,axcolmag_histy
    return
######################################################################################################################
# Colour-Magnitude Scatter plot with marginal histograms with objects weighted by 1/Vmax
def plot_col_mag_withvmax(dat,regions):
    def flatten(l):        # limits of data to use in histogram()
        return [item for sublist in l for item in sublist]

    range=[[-0.1,1.3],[-23.99,-14]] # range to use for colour and absolute magnitude limits (both selection and plotting limits)
    nbin=100                       # number of bins per dimension
    sigma=1                        # gaussian smoothing in units of bin width
    levels= [16000,32000,64000,128000,256000,512000,1024000,2048000] # contour levels in points per bin
    ABSMAG_BRIGHT=-23.0
    ABSMAG_FAINT=-22.0

    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.0, hspace=0.0)
    # Create the Axes.
    axcolmag = fig.add_subplot(gs[1, 0])
    axcolmag_histx = fig.add_subplot(gs[0, 0], sharex=axcolmag)
    axcolmag_histy = fig.add_subplot(gs[1, 1], sharey=axcolmag)

    axcolmag_histx.tick_params(axis="x", labelbottom=False) #suppress default axis labels
    axcolmag_histy.tick_params(axis="y", labelleft=False)

    

    #Marginal Colour distribution
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask) & (dat['ABSMAG_RP1']<ABSMAG_FAINT) & (dat['ABSMAG_RP1']>ABSMAG_BRIGHT)
        hist,x=np.histogram(dat['REST_GMR_0P1'][mask],bins=nbin,range=range[0],weights=1.0/dat['vmax'][mask])
        bin_centres= 0.5*(x[1:]+x[:-1])
        axcolmag_histx.plot(bin_centres,hist,color=Sel['col'])
        ymax=1.1*hist.max()
        axcolmag_histx.set(ylim=[0,ymax])

    #Marginal absolute magnitude distribution
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask) & (dat['ABSMAG_RP1']<ABSMAG_FAINT) & (dat['ABSMAG_RP1']>ABSMAG_BRIGHT)
        hist,x=np.histogram(dat['ABSMAG_RP1'][mask],bins=nbin,range=range[1],weights=1.0/dat['vmax'][mask])
        bin_centres= 0.5*(x[1:]+x[:-1])
        axcolmag_histy.plot(hist,bin_centres,color=Sel['col'])
        xmax=1.1*hist.max()
        axcolmag_histy.set(xlim=[0,xmax])
        
    #2D scatter plots
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        #Object selection mask
        smask=(dat['Z'] > Sel['zmin']) & (dat['REST_GMR_0P1']>-0.1)  & (dat['REST_GMR_0P1']<1.4) & (dat['rmag']>Sel['bright']) & (dat['rmag'] < Sel['faint'])
        regmask = (dat['reg']==reg)
        #sample selection
        mask = (regmask) & (smask) & (dat['ABSMAG_RP1']<ABSMAG_FAINT) & (dat['ABSMAG_RP1']>ABSMAG_BRIGHT)
        axcolmag.scatter(dat[mask]['REST_GMR_0P1'],dat[mask]['ABSMAG_RP1'], marker='.', linewidths=0,s=0.25,color=Sel['col'],alpha=0.1,label='South')
        #Contours of the point distribution in the scatter plots
        extent=flatten(range)  # flattened version to use in contour()
        counts,x,y=np.histogram2d(dat['REST_GMR_0P1'][mask],dat['ABSMAG_RP1'][mask],bins=nbin,range=range,weights=1.0/dat['vmax'][mask])
        counts=gaussian_filter(counts,sigma)
        axcolmag.contour(counts.transpose(),extent=extent,levels=levels, colors=Sel['col'])



    range[1].reverse()  #flip the Absolute magnitude axes direction
    axcolmag.set(xlabel='$M_g - M_r$', ylabel='$M_r -5 \log h$',xlim=range[0],ylim=range[1])
    Sel=selection('N')
    colN=Sel['col']
    Sel=selection('S')
    colS=Sel['col']
    axcolmag.legend(['North','South'],loc='upper left',labelcolor=[colN,colS])
    
    del mask,smask,regmask # tidy up
    del axcolmag,axcolmag_histx,axcolmag_histy
    return
####################################################################################################################### All-sky maps 
#  see https://notebook.community/desihub/desiutil/doc/nb/SkyMapExamples for examples of how to make plots like these
def sky_plot(dat,regions):

    # All-sky scatter plot with galactic plane and ecliptic marked
    ax= init_sky()
    for reg in regions:
        regmask = (dat['reg']==reg)
        Sel=selection(reg) # define selection parameters for this region
        p = ax.scatter(ax.projection_ra(dat['RA'][regmask]),ax.projection_dec(dat['DEC'][regmask]),color=Sel['col'],s=0.25, marker='.', linewidths=0)


    # healpix source density map in healpix's of less than max_bin_area sq degrees
    ax= plot_sky_binned(dat['RA'] ,dat['DEC'], plot_type='healpix', max_bin_area=4.0, verbose=True)
    return
######################################################################################################################
def sky_plot_jack(dat):
    colour = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'purple', 'orange', 'brown', 'pink', 'lime', 'teal', 'lavender', \
    'turquoise', 'gold', 'darkblue', 'darkgreen', 'darkred', 'darkcyan', 'darkmagenta', 'forestgreen', 'lightblue', 'lightgreen', 'lightcoral',  'lightpink', 'lightskyblue', 'lightgray']
    

    # All-sky scatter plot with galactic plane and ecliptic marked
    mask=(dat['ijack']!=-999) #exclude those set to -999 which means unassigned
    njack=1+np.max(dat["ijack"][mask])-np.min(dat["ijack"][mask]) #number of jackknife samp
    ax= init_sky()
    for ijack in range(0, njack ):  
        mask = (dat['ijack']==ijack)
        p = ax.scatter(ax.projection_ra(dat['RA'][mask]),ax.projection_dec(dat['DEC'][mask]),color=colour[ijack],s=0.25, marker='.', linewidths=0)  
    # PLot the problematic points that have not been assigned to a jackkife region
    mask=(dat['ijack']==-999)
    p = ax.scatter(ax.projection_ra(dat['RA'][mask]),ax.projection_dec(dat['DEC'][mask]),color='black',s=20.0, marker='X', linewidths=0)  
    return
###################################################################################################    
#Cone plot
def cone_plot(dat,regions):
    #This is hardwired to produce two particular cone plots but could be adapted
    for reg in regions:
        regmask = (dat['reg']==reg)
        Sel=selection(reg) # define selection parameters for this region
        # Selection to remove stars and impose chosen magnitude limit
        if (reg=='S'):
            mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['rmag'] < Sel['faint']) &  (dat['DEC'] > -5)  &  (dat['DEC'] < 0)  &  (dat['RA'] > 180.0) &  (dat['RA'] < 280.0)
        else:
            mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['rmag'] < Sel['faint']) &  (dat['DEC'] > 30.0)  &  (dat['DEC'] < 35.0)  &  (dat['RA'] > 180.0) &  (dat['RA'] < 280.0)
        
        print(np.count_nonzero(mask),' points in the cone plot')
        offset=+1.5*np.pi/2.0
        x=dat[mask]['Z']*np.sin(np.radians(dat[mask]['RA'])+offset)
        y=dat[mask]['Z']*np.cos(np.radians(dat[mask]['RA'])+offset)
        plt.axis('equal')
        plt.axis('off')
        plt.scatter(x,y,color=Sel['col'],s=0.25, marker='.', linewidths=0)
        ra_min=dat[mask]['RA'].min()
        ra_max=dat[mask]['RA'].max()
        ra=np.linspace(ra_min,ra_max,100)
        z_max=dat[mask]['Z'].max()
        frame_x=np.concatenate((np.array([0]),z_max*np.sin(np.radians(ra)+offset),np.array([0])))
        frame_y=np.concatenate((np.array([0]),z_max*np.cos(np.radians(ra)+offset),np.array([0])))
        plt.plot(frame_x,frame_y,color='black')
        plt.show()
    del x,y,ra # tidy up
    return
######################################################################################################################
#Plot 1/vmax Luminosity function
#with different plotting options
def lumfun_vmax(dat,regions, bandmag='ABSMAG_RP1', band='R', plot=True, saveplot=False, binwidth=0.25, ratio=False, Veff=False):
    
    eps=1.0e-10 #used to avoid divide by zero
    bins = np.arange(-24.0, -11.1, binwidth)
    bin_cen = (bins[:-1] + bins[1:]) / 2.0
    for reg in regions:
        Sel=selection(reg) # define selection parameters for this region
        regmask = (dat['reg']==reg)
        # Selection to remove stars and impose chosen magnitude limit
        mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['Z'] < Sel['zmax']) \
        & (dat['rmag'] < Sel['faint'])  & (dat['rmag'] > Sel['bright']) 
        # A weighted histogram using the combined systematics and completeness weight and 1/Vmax
        weight=np.copy(dat['WEIGHT'])# just to intialize 
        if Veff:
            print('Using Veff_max rather than Vmax in LF estimate.')
            weight[mask]=dat['WEIGHT'][mask]/(binwidth*dat['veff_max'][mask]*Sel['f_ran'])
        else:
            weight[mask]=dat['WEIGHT'][mask]/(binwidth*dat['vmax'][mask]*Sel['f_ran'])
        phi,binr=np.histogram(dat[bandmag][mask], bins=bins,  weights=weight[mask])
        
        log_phi=np.log10(np.maximum(phi,eps))

        # compute jackknife errors if jackknife indices exist
        print("jackknife index range:",np.min(dat["ijack"][mask]),np.max(dat["ijack"][mask]))
        njack=1+np.max(dat["ijack"][mask])-np.min(dat["ijack"][mask]) #number of jackknife samples
        print('number of jackknife samples:',njack,' for region:',reg)
        phi_jack=np.zeros((njack,log_phi.size),dtype=float)# array in which to store all the jackknife estimates
        if (njack>1):  #Only do jackknife errors if we have jackknife indices otherwise default to Poisson errors
           for ijack in range(np.min(dat["ijack"][mask]), np.max(dat["ijack"][mask]) + 1):
               jjack=ijack-np.min(dat["ijack"][mask]) #offset index so this region starts at zero
               #print("ijack",ijack,'jjack',jjack)
               jkmask= (dat["ijack"]!=ijack) & mask  # exlcude one region
               phi_jack[jjack,:],binr=np.histogram(dat[bandmag][jkmask], bins=bins,  weights=weight[jkmask])
               phi_jack[jjack,:]=phi_jack[jjack]*njack/(njack-1)  # rescale to account for the missing region 
               log_phi_jack=np.log10(np.maximum(phi_jack[jjack,:],eps))
               #plt.plot(bin_cen,log_phi_jack) # plot each jackknife estimate in turn
           phi_mean=np.mean(phi_jack,axis=0) #compute the mean over the jackknife samples
           phi_err=np.std(phi_jack,axis=0)*np.sqrt(njack-1)  # jackknife rescaling factor as np.std()'s default is ddof=0
           log_phi_jack=np.log10(np.maximum(phi_mean,eps))
           #plt.plot(bin_cen,log_phi_jack,linewidth=3) # plot mean of jackknife estimates
           log_phi_low=np.log10(np.maximum(phi-phi_err,eps))
           log_phi_hi=np.log10(np.maximum(phi+phi_err,eps))
           
            
        else:
           print("Estimating Poisson Errors as Jackknife indices not set")
           weight=weight**2 # For Poisson error= sqrt(Sum w^2)
           phi_err,binr=np.histogram(dat[bandmag][mask], bins=bins,  weights=weight[mask])
           phi_err=np.sqrt(phi_err)
           log_phi_low=np.log10(np.maximum(phi-phi_err,eps))
           log_phi_hi=np.log10(np.maximum(phi+phi_err,eps))
        
        
        # Reference Schehcter function from Sam's draft paper Table 1
        phi_star=10.0**(-2.13)
        alpha=-1.36
        mstar=-21.10
        l_lstar= 10.0**(0.4*(-bin_cen+mstar))
        phi_sch= phi_star * l_lstar**(1.0+alpha) * np.exp(-l_lstar) *(np.log(10.0)/2.5)
        log_phi_sch = np.log10(phi_sch)
        
        # plot and later superpose the reference
        if plot and not ratio:
            plt.fill_between(bin_cen,log_phi_low,log_phi_hi,color=Sel['col'],alpha=0.5)

        # plot relative to the reference    
        if plot and ratio:
            plt.fill_between(bin_cen,log_phi_low-log_phi_sch,log_phi_hi-log_phi_sch,color=Sel['col'],alpha=0.5)
    

    
    if plot and not ratio:
        plt.plot(bin_cen,log_phi_sch, label='Reference Schechter Function')
        plt.xlabel('$M_r - 5 log h$')
        plt.ylabel('$log_{10} \phi(M_r)\quad  [mag^{-1} (Mpc/h)^{-3}]$')
        plt.xlim([-26,-7.5])
        plt.ylim([-7,1.0])
        #Add label to the legend to indicate what sort of error estimate was used
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='black', linestyle='None'))
        if (njack>1):labels.append("Jackknife errors")
        else: labels.append("Poisson errors")
       
        
        plt.legend(handles, labels)
        plt.show()
        if saveplot:
            spath = './graphs/global_LF.pdf'
            plt.savefig(spath)

    if plot and ratio:
        plt.xlabel('$M_r - 5 log h$')
        plt.ylabel('$log_{10} \phi(M_r)/\phi_{ref}(M_r) \quad  [mag^{-1} (Mpc/h)^{-3}]$')
        plt.xlim([-25,-9])
        plt.ylim([-0.3,+0.5])
        plt.show()
        

    del weight,log_phi_sch,l_lstar,phi_sch,phi,phi_err #tidy up
    return log_phi,log_phi_low,log_phi_hi,bin_cen   # return the LF and error band of the estimate from the last region
######################################################################################################################
#Plot SWML Luminosity function   (Uses method from Eftathiou, Ellis and Petersen 1988 MNRAS 232,431 [EEP])
def lumfun_swml(dat,regions,log_phi_guess,magbins):
    
    #
    extra_plots=False
    
    # Compute luminosity density of the input LF which will be used for the normalization
    lumden=np.sum((10.0**(-0.4*magbins)*(10.0**log_phi_guess)))
    
    # Array for denominator in EEP expression Eqn(2.12), which is a kind of effective volume
    denominator=np.zeros(magbins.size)
    # Array for the faint and bright absolute magnitude limits at each galaxy redshift
    absmag_faint=np.zeros(dat['Z'].size) 
    absmag_bright=np.zeros(dat['Z'].size) 
    # Array for the cumulative LF brighter than the faint limit at each galaxy redshift
    phi_cuml_gal=np.zeros(dat['Z'].size)   
    log_phi_cuml_gal=np.zeros(dat['Z'].size)   # and for the log version
    hramp=np.zeros(dat['Z'].size) #array for the ramp function h(L_min<L<L_max)
    
    # Various bin definitions: We need the bin edges that correspond to specified bin centres
    n=magbins.size-1 #index of last element in magbins[]
    first_edge=np.array(1.5*magbins[0]-0.5*magbins[1],ndmin=1)
    last_edge=np.array(1.5*magbins[n]-0.5*magbins[n-1],ndmin=1)
    magbins_faint_edge=np.concatenate(((0.5*(magbins[1:]+magbins[:-1])),last_edge),axis=None) #faint edge of bins used in cumulative LF
    bin_edges=np.concatenate((first_edge,(0.5*(magbins[1:]+magbins[:-1])),last_edge),axis=None) #bin_edges used by np.histogram()
    bin_width=magbins[1]-magbins[0]
    halfbin=0.5*bin_width
    print('bin width',bin_width)
    
    plt.plot(magbins,log_phi_guess,linewidth=0.75,color='black',label='$1/V_{max}$') # plot the starting LF for comparison
    #Loop over regions making separate estimates
    for reg in regions:
        Sel=selection(reg) #load selection parameters and k-corrections for this region
        kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg)
        
        regmask=(dat['reg']==reg) #select objects in this region
        mask = (regmask) & (dat['Z'] > Sel['zmin']) & (dat['Z'] < Sel['zmax']) \
        & (dat['rmag'] < Sel['faint'])  & (dat['rmag'] > Sel['bright']) #satisfying the selection cuts
        print('number in full sample:',np.count_nonzero(mask))
  
        # Compute the count of galaxies in each absolute magnitude bin -- remains fixed in each iteration
        count,bins = np.histogram(dat['ABSMAG_RP1'][mask], bins=bin_edges, weights=dat['WEIGHT'][mask])
                             
        # Compute the absolute faint magnitude limit at the redshift of the galaxy using its k-correction
        absmag_faint[mask] = ABSMAG(Sel['faint'],dat['Z'][mask],dat['REST_GMR_0P1'][mask],kcorr_r,Sel['Qevol'])  
        # Compute the absolute bright magnitude limit at the redshift of the galaxy using its k-correction
        absmag_bright[mask] = ABSMAG(Sel['bright'],dat['Z'][mask],dat['REST_GMR_0P1'][mask],kcorr_r,Sel['Qevol'])  
        if extra_plots:
            plt.scatter(dat['Z'][mask],absmag_faint[mask], marker='.', linewidths=0, s=0.25, alpha=0.5,label='faint') # quick look to see if reasonable
            plt.scatter(dat['Z'][mask],absmag_bright[mask], marker='.', linewidths=0, s=0.25, alpha=0.5,label='bright') # quick look to see if reasonable
            plt.xlabel('z')
            plt.ylabel('Absmag (faint/bright limit)')
            plt.legend()
            plt.show()
    
        # Start iteration begining with the guess that was supplied 
        phi1=10.0**log_phi_guess   # first guess
        phi0=np.zeros(phi1.size)   # array for previous iteration
        eps=0.1*bin_width # a value smaller than the magnitude bin width. Used to select individual bins and avoid rounding error problems
        tol=0.01 # fractional tolerance allowed in LF estimate
        tolabs=1.0e-09 #additional absolute tolerance allowed in LF estimate
        i=-1 # iteration counter
        imax=12
        while  ( (np.any(np.abs(phi1-phi0)>tol*phi1+tolabs)) & (i<imax)):   #iterate until all bins have converged to tolerance tol
            i+=1
            phi0[:]=phi1 #update last iteration to current estimate by copying values 
            phi_cuml= np.cumsum(phi1)*bin_width #cumulative luminosity above faintest bin edge
            log_phi_cuml=np.log10(np.maximum(phi_cuml,1.0e-10))
            # now need to interoplate to the faint & bright absmag limit at the redshift of each selected galaxy
            #linear interpolation
            phi_cuml_gal[mask]=np.interp(absmag_faint[mask], magbins_faint_edge,phi_cuml) \
                              -np.interp(absmag_bright[mask],magbins_faint_edge,phi_cuml)     
           
            
            log_phi_cuml_gal[mask]=np.log10(phi_cuml_gal[mask]) 

            for absmag in magbins: # loop over the LF bins
                magmask= (np.abs(absmag-magbins)<eps) #mask to select the single LF absmag bin we are going to update
                
                #mask to select galaxies in each range of the H-function!!!
                hramp=np.zeros(dat['Z'].size) # default value
                halfbin=0.5*bin_width
                galmask= (mask) & (absmag<absmag_faint+halfbin) & (absmag>absmag_faint-halfbin) 
                hramp[galmask]= (absmag_faint[galmask]-absmag)/bin_width+0.5 #faint ramp from 0 to 1
                galmask= (mask) & (absmag<absmag_faint-halfbin) & (absmag>absmag_bright+halfbin) 
                hramp[galmask]= 1.0 #plateau at 1
                galmask= (mask) & (absmag<absmag_bright+halfbin) & (absmag>absmag_bright-halfbin) 
                hramp[galmask]= (absmag-absmag_bright[galmask])/bin_width+0.5 #bright ramp from 1 to 0
                galmask= (mask) & (absmag<absmag_faint+halfbin)# & (absmag>absmag_bright-halfbin) #mask to select all galaxies to sum over
                        #i.e. all those at redshifts for which the bright/faint limit is brighter/fainter than the bin value
                denominator[magmask]=np.sum((dat['WEIGHT'][galmask]*hramp[galmask]/phi_cuml_gal[galmask])) #denominator of EEP Eqn(2.12)
                if ((count[magmask]>0) & (denominator[magmask]>0) ):
                    phi1[magmask]=count[magmask]/(bin_width*denominator[magmask]) # EEP Eqn(2.12) for bins that have data
                    
            fac=lumden/np.sum((10.0**(-0.4*magbins)*phi1)) # luminosity density ratio
            print('region:',reg,': luminosity density correction factor:',fac)
            phi1=phi1*fac
            log_phi=np.log10(np.maximum(phi0,1.0e-10))
           
    
        log_phi=np.log10(np.maximum(phi1,1.0e-10))
        print('converged after',i,' iterations or reached maximum',imax)
        plt.plot(magbins,log_phi,label='SWML i={} ({})'.format(i, reg),color=Sel['col'])
        
    # Reference Schehcter function from Sam's draft paper Table 1
    phi_star=10.0**(-2.13)
    alpha=-1.36
    mstar=-21.10
    l_lstar= 10.0**(0.4*(-magbins+mstar))
    phi_sch= phi_star * l_lstar**(1.0+alpha) * np.exp(-l_lstar) *(np.log(10.0)/2.5)
    log_phi_sch = np.log10(phi_sch)
    plt.plot(magbins,log_phi_sch, label='Reference Schechter Function')

    plt.xlabel('$M_r - 5 log h$')
    plt.ylabel('$\phi(M_r)\quad  [mag^{-1} (Mpc/h)^{-3}]$')
    plt.xlim([-26,-7.5])
    plt.ylim([-7.0,1.0])
    plt.legend()
    
    spath = './graphs/SWML_LF.pdf'
    plt.savefig(spath)
    plt.show()

    del phi0,phi1,count,absmag_faint,phi_cuml,denominator,mask,regmask,magmask,galmask
    return log_phi,magbins
####################################################################################################################################
#  Generate a random catalogue with no clustering and specified LF
#  **Haven't checked that it get things right to within one bin width**
def makefake(reg,nran):
    Sel=selection(reg) 
    #Generate points in spherical shell with chosen [zmin,zmax] 
    rmin=cosmo.comoving_distance(Sel['zmin']).value
    rmax=cosmo.comoving_distance(Sel['zmax']).value
    Qevol=Sel['Qevol']
    print('Generating Randoms')
    # start by generating in a cube
    x=2.0*rmax*(np.random.rand(nran)-0.5)
    y=2.0*rmax*(np.random.rand(nran)-0.5)
    z=2.0*rmax*(np.random.rand(nran)-0.5)
    r=np.sqrt(x**2+y**2+z**2)
    mask = (r<rmax) & (r>rmin)  #keep if within the shell

    #find redshift for each retained position
    def distance(z,aux):
        distance=cosmo.comoving_distance(z).value
        return distance


    zmins=Sel['zmin']*np.ones(r.size)
    zmaxs=Sel['zmax']*np.ones(r.size)
    aux=np.zeros(r.size)
    zred=np.zeros(r.size)
    eps=0.00001
    zred[mask]=root_itp(distance,aux[mask],r[mask],eps,zmins[mask],zmaxs[mask])

    #Generate absolute magnitudes from a Schechter function
    bin_cen=np.linspace(-24.0,-10.0,4000)
    bin_width=bin_cen[1]-bin_cen[0]
    # McNaught-Roberts reference Schechter function from Section 3.1
    phi_star=10.0**(-2.01)
    alpha=-1.25
    mstar=-20.89
    l_lstar= 10.0**(0.4*(-bin_cen+mstar))
    phi_sch= phi_star * l_lstar**(1.0+alpha) * np.exp(-l_lstar) *(np.log(10.0)/2.5)
    #phi_sch=np.exp(-32.0*(np.log10(l_lstar*10.0))**2)  # Gaussian LF
    phi_cuml= np.cumsum(phi_sch)*bin_width # cumulative luminosity above faintest bin edge
    print('phi cumulative range',phi_cuml[0],phi_cuml[-1])
    log_phi_cuml=np.log10(np.maximum(phi_cuml,1.0e-10))


    
    def phi_cumulative(absmag,aux):
         log_phi_cumulative=np.interp(absmag,bin_cen,log_phi_cuml)#1/2 bin shift?  but bins tiny :) 
         phi_cumulative=10.0**log_phi_cumulative        
         return phi_cumulative
 



    # target cumulative phi uniformly distributed over the full range
    phi_cuml_target=np.random.rand(nran)*(phi_cuml[-1]-phi_cuml[0]) + phi_cuml[0]
    phi_cuml0=-24.0*np.ones(r.size) # bounds on roots
    phi_cumlf=-10.0*np.ones(r.size)
    eps=eps*phi_cuml[-1]
    absmag=root_itp(phi_cumulative,aux,phi_cuml_target,eps,phi_cuml0,phi_cumlf)



    #Compute apparent magnitudes assuming fixed colour k-corretion
    kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys='N')
    kcorr_g  = DESI_KCorrection(band='G', file='jmext', photsys='N')
    #rest_GMR=1.0*np.ones(r.size) #assign rest frame colour
    rest_GMR=0.5+0.25*np.random.normal(size=r.size) #assign rest frame colour
    rmag=np.zeros(r.size)
    gmr=rest_GMR-kcorr_r.k(zred,rest_GMR)+kcorr_g.k(zred,rest_GMR)
    # ABSMAG_R= appmag -DMOD  -kcorr_r.k(z, rest_GMR) +Qevol*(z-0.1) 
    def APPMAG(absmag,z,rest_GMR,kcorr_r,Qevol):
        DMOD=25.0+5.0*np.log10(cosmo.luminosity_distance(z).value) 
        APPMAG =  absmag +DMOD  +kcorr_r.k(z, rest_GMR) -Qevol*(z-0.1)
        return APPMAG
    
    rmag[mask]=APPMAG(absmag[mask],zred[mask],rest_GMR[mask],kcorr_r,Sel['Qevol'])    

    #Reject objects outside apparent magnitude range and save/return remaining catalogue
    fmask= (mask) & (rmag<Sel['faint']) & (rmag>Sel['bright'])

    plt.hist(absmag[fmask],bins=20)
    plt.xlabel('absmag')
    plt.show()
    plt.hist(zred[fmask],bins=20)
    plt.xlabel('z')
    plt.show()

    plt.scatter(zred[fmask],rmag[fmask])
    plt.ylabel('absmag')
    plt.xlabel('z')
    plt.show()

    #

    reg=np.full(zred.size,'N')
    weight=np.full(zred.size,1.0)
    dat = Table([absmag[fmask], zred[fmask], rest_GMR[fmask], rmag[fmask], gmr[fmask], reg[fmask], weight[fmask]], names=('ABSMAG_RP1', 'Z', 'REST_GMR_0P1', 'rmag', 'gmr_obs', 'reg', 'WEIGHT'), meta={'name': 'mock data'})
    dat.info('stats')
    return dat
#################################################################################################################################################################################################################
#Computes a V_eff for each galaxiy that can be used in place V_max in LF and other weighted estimates.

# The basic idea is V_eff= integral  Delta(z) dV/dz  dz  where the overdensity Delta(z) is estimated 
# by making clones of galaxies and redistributing them withing their V_max volume to define a smooth dN/dz
# from which can be compared with the actual dN/dz to define the run overdensity with redshift.
#

def compute_veff(dat,regions):
    #Seemed to work better (larger p-values) before the weigths dat['WEIGHT'] were inserted.
    #But actually that should happen as it is the weigthed Veff/Veff,max that should be uniform
    #not the unweighted one we are testing.

    # To Do:
    #    i) A rebinned density versus redshift plot to monitor convergence 
    #   ii) A measure of convergence
    #

    #We are using V as the radial coordinate with V=0 corresponding to vmin_sample.
    #But the dat['v'] value of each galaxies is relative to its own dat['vmin'] which for some of the bright galaxies
    #can be greater than vmin_sample.
    #So for our V coordinate we have to add back on dat['vmin'] and subtract vmin_sample
    # Then we also have to take account dat['vmin'] >vmin_sample when placing their clones and computing their final v_eff,max.

    # Set up arrays into which we will put the values per galaxy
    veff=np.zeros(dat['Z'].size)
    veff_max=np.zeros(dat['Z'].size)
    ratio=np.zeros(dat['Z'].size)
    #dat.add_column(Column(name='veff', data=veff))
    #dat.add_column(Column(name='veff_max', data=veff_max))

    nsamp=64.0 #over sampling factor
    nbins=2000 #number of volume bins
    nzbins=20 #number of redshift bins (just for plots)

    for reg in regions:           #Do regions separately (Would have to define volume bins differently to merge)
      print('starting region ',reg)
      regmask=(dat['reg']==reg)#mask to select objects in specified region

      Sel=selection(reg)
      #minimum vmin for the whole selected sample. We just consider volume above this vmin and work in bins of that covering 0 to vmax-vmin
      vmin_sample=Sel['area']*(4.0*np.pi/3.0)*(cosmo.comoving_distance(Sel['zmin']).value)**3 
      vmax_sample=Sel['area']*(4.0*np.pi/3.0)*(cosmo.comoving_distance(Sel['zmax']).value)**3 - vmin_sample
      vbins=np.linspace(0.0,vmax_sample,nbins) # volume bin edges
      vbin_width=vbins[1]-vbins[0] #bin width
      print('vbin_width:',vbin_width*1.0e-09,vmax_sample*1.0e-09/(nbins-1),'(Gpc/h)^3')  
      vbin_cen=(vbins[:-1]+vbins[1:])/2.0 #volume bin centres
      Delta_vbin=np.ones(vbin_cen.size)  #Overdensity in each volume bin initialised to uniform density
      Delta=np.ones(dat['Z'].size)       #Overdensity for each galaxy initialised to uniform density
      v=np.zeros(dat['Z'].size)          #initialise array for random volume values for each galaxy 
      zbins=np.linspace(Sel['zmin'],Sel['zmax'],nzbins) # zbin edges (used for plots)
      zbin_cen=(zbins[:-1]+zbins[1:])/2.0 #zbin centres

      # For top axis of some plots we set up a grid of redshift tick marks and compute their volume coordinate
      zgrid=np.linspace(0.1,Sel['zmax'],5)
      zgrid_strings = ["%.1f" % number for number in zgrid] #convert to strings rounded to 1 decimal place
      vgrid=Sel['area']*(4.0*np.pi/3.0)*(cosmo.comoving_distance(zgrid).value)**3 - vmin_sample

      #Check that data values are all in expected ranges
      vcheck=  vmax_sample-dat['v'][regmask]
      if (np.min(vcheck)<0): print('ERROR: object with V>V_max')
      vcheck=  vbins[nbins-1]- (dat['v'][regmask]+dat['vmin'][regmask]-vmin_sample)
      if (np.min(vcheck)<0): print('ERROR: object with V+vmin-vmin_sample> largest vbin edge')
      vcheck=  (dat['v'][regmask]+dat['vmin'][regmask]-vmin_sample)
      if (np.min(vcheck)<0): print('ERROR: object with V+vmin<vmin_sample')
      vcheck=  (dat['vmin'][regmask]-vmin_sample)
      if (np.min(vcheck)<0): print('ERROR: object with vmin<vmin_sample')
      zcheck=  (dat['zmin'][regmask]-Sel['zmin'])
      if (np.min(zcheck)<0): print('ERROR: object with zmin<zmin_sample')

      # Plot the number of galaxies per volume  (and redshift) bin  (This will increase towards low volume as there we see fainter galaxies)
      vhist_dat,vbinss=np.histogram((dat['v'][regmask]+dat['vmin'][regmask]-vmin_sample),bins=vbins,weights=dat['WEIGHT'][regmask]) #data histogram in volume coordinate bins
      zhist_dat,zbinss=np.histogram(dat['Z'][regmask],bins=zbins,weights=dat['WEIGHT'][regmask])



      # Set up redshift labels for the top axis
      ax = host_subplot(211)
      ax2 = ax.twin()
      ax2.set_xlabel('$z$')    
      ax2.xaxis.set_label_position('top') 
      ax2.set_xticks(vgrid,labels=zgrid_strings)   
      ax.plot(vbin_cen,np.log10(np.maximum(vhist_dat,1.0e-10)))
      plt.xlim([0,vmax_sample])  
      plt.ylim([0,6.0])
      plt.xlabel('$V /(Mpc/h)^3$')
      plt.ylabel('$log_{10}$ N(V)')
      ax3 = host_subplot(212)
      ax3.plot(zbin_cen,np.log10(np.maximum(zhist_dat,1.0e-10)))
      plt.xlabel('$z$')
      plt.ylabel('$log_{10}$ N(z)')
      plt.show()

      n=np.zeros(dat['Z'].size, dtype=int)        #initialise number of clones of each galaxy to zero
      iv=np.zeros(dat['Z'].size, dtype=int)       
      iv[regmask]=np.digitize(dat['v'][regmask]+dat['vmin'][regmask]-vmin_sample,vbins)-1 #find the volume bin index for each galaxy  

      # Set up redshift labels for the top axis
      ax = host_subplot(111)
      ax2 = ax.twin()
      ax2.set_xlabel('$z$')    
      ax2.xaxis.set_label_position('top') 
      ax2.set_xticks(vgrid,labels=zgrid_strings)  


      for j in range(10): #loop until Deltas converge within Poisson Errors???? -- just loops 6 times for now
        nmean=nsamp*dat['WEIGHT']/Delta #mean number of times to redistribute each galaxy

        n[regmask]=np.floor(nmean[regmask]).astype(int) #Round down to the nearest integer
        mask= (np.random.uniform(Delta.size) < nmean-n) #Choose with probability equal to how much nmean exceeds n those for which n is tobe increased by 1
        cmask=mask&regmask #combined mask for region and those needing extra increment 
        n[cmask] += 1 # increment if nmean>n with probability such that <n>=nmean for those in the region
        nmax=n.max() #  the maximum number of times any galaxy will be cloned
        vhist=np.zeros(vhist_dat.size) # initialize vhist histogram of the number of clones in each volume bin
        for i in range(nmax): #loop over the maximum number of times we need to clone any galaxy
            mask= (i<n) #add a clone of all the galaxies that need i or more clones (remember i starts from 0)
            cmask=mask&regmask #combined region and those needing incrementing mask
            nclones=np.count_nonzero(cmask) #this is the total number of galaxies we are cloning on this pass
            # print('pass:',i+1,'of',nmax,' extra clones added:',nclones)
            v[cmask]=dat[cmask]['vmax']*np.random.uniform(size=nclones)# distribute each clone uniformly within its vmax
            hist,vbinss=np.histogram(v[cmask]+dat[cmask]['vmin']-vmin_sample,bins=vbins) #bin the clones in volume coordinate bins
            vhist += hist  # increment the histogram with the clones from this pass
        # Now we have cloned and binned all the galaxies we can compute the overdensity Delta in each volume coordinate bin    
        binmask = (vhist>0)  #mask to avoid divide by zero in any bin with no clones 
        Delta_vbin[binmask]=nsamp*vhist_dat[binmask]/vhist[binmask] #overdensity relative to clones
        Delta_vbin=Delta_vbin/np.mean(Delta_vbin)  #Renormalize the mean which is necessary as <nmean>.ne.nsamp as <1/Delta>.ne.1
        print('RMS Delta:',np.sqrt(np.mean(((Delta_vbin-np.mean(Delta_vbin))**2))))
        plt.plot(vbin_cen,Delta_vbin,label=str(j))# plot overdensity at this iteration  
        Delta[regmask]=Delta_vbin[iv[regmask]] # update each galaxy's overdensity (for galaxies in the region)

        #Form the cumulative sum of V_eff up to the bin edges vbin. This is exactly what we need for interpolation.
        veff_bin=np.concatenate((np.array([0.0]),np.cumsum(Delta_vbin)*vbin_width))# cumulative distribution including 0 at first bin edge

        # The first two expressions below would be right if dat['vmin']= vmin_sample but for those with dat['vmin']>vmin_sample we have to make a correction
        veff[regmask]    =np.interp(dat['v'][regmask]   +dat['vmin'][regmask]-vmin_sample, vbins,veff_bin) #For each galaxy interpolate veff from the cumulative binned values
        veff_max[regmask]=np.interp(dat['vmax'][regmask]+dat['vmin'][regmask]-vmin_sample, vbins,veff_bin) #The same for each galaxies Vmax value
        mask=(dat['vmin']>vmin_sample)
        cmask=mask & regmask  #combined mask for those in the region needing their v_eff correcting as dat['vmin']>vmin_sample
        veff[cmask]    =veff[cmask]    - np.interp(dat['vmin'][cmask]-vmin_sample, vbins,veff_bin)
        veff_max[cmask]=veff_max[cmask]- np.interp(dat['vmin'][cmask]-vmin_sample, vbins,veff_bin)  
        #
        plt.scatter(dat['v'][regmask],veff[regmask]-dat['v'][regmask], s=0.25)
        # Go back and do next iteration    
      #
      #Finish off the plot to see how well convergence has worked
      plt.xlim([0,vmax_sample])
      plt.ylabel('$V_{eff} -V$')
      plt.xlabel('$V$')
      ax.legend(loc='upper left')
      plt.show()

    # Store the veff and veff_max values that are returned and look at their stats
    dat['veff']=veff
    dat['veff_max']=veff_max
    dat.info('stats')  

    # This was meant to be test of V_eff/V_eff,max is more uniform than V/V_max but to do that it needs to take account the weights but currently does not.
    ratio[regmask]=dat['v'][regmask]/dat['vmax'][regmask]
    plt.hist(ratio[regmask], bins=nbins, label='$V/V_{max}$',histtype='step')
    print('V/V_max uniformity',stats.kstest(ratio[regmask],'uniform'))
    ratio[regmask]=dat['veff'][regmask]/dat['veff_max'][regmask]
    plt.hist(ratio[regmask], bins=nbins, label='$V_{eff}/V_{eff,max}$',histtype='step')
    print('V_eff/V_eff,max uniformity',stats.kstest(ratio[regmask],'uniform'))
    plt.xlabel('$V_{eff}/V_{eff,max}$')
    plt.xlim([0,1.0])
    plt.legend()
    plt.show()

   
    return
#################################################################################################################################################
#Computes a V_eff for each galaxiy that can be used in place V_max in LF and other weighted estimates.

# The basic idea is V_eff= integral  Delta(z) dV/dz  dz  where the overdensity Delta(z) is estimated 
# by making clones of galaxies and redistributing them withing their V_max volume to define a smooth dN/dz
# from which can be compared with the actual dN/dz to define the run overdensity with redshift.
# This version using log binning

def compute_veff_logspacing(dat,regions):
    #Seemed to work better (larger p-values) before the weigths dat['WEIGHT'] were inserted.
    #But actually that should happen as it is the weigthed Veff/Veff,max that should be uniform
    #not the unweighted one we are testing.

    # To Do:
    #    i) A rebinned density versus redshift plot to monitor convergence 
    #   ii) A measure of convergence
    #

    #We are using V as the radial coordinate with V=0 corresponding to vmin_sample.
    #But the dat['v'] value of each galaxies is relative to its own dat['vmin'] which for some of the bright galaxies
    #can be greater than vmin_sample.
    #So for our V coordinate we have add back on dat['vmin'] and subtract vmin_sample
    # Then we also have to take account dat['vmin'] >vmin_sample when placing their clones and computing their final v_eff,max.

    # Set up arrays into which we will put the values per galaxy
    veff=np.zeros(dat['Z'].size)
    veff_max=np.zeros(dat['Z'].size)
    ratio=np.zeros(dat['Z'].size)
    dat.add_column(Column(name='Delta', data=np.ones(dat['Z'].size) )) #Overdensity for each galaxy initialised to uniform density 
    #dat.add_column(Column(name='veff', data=veff))
    #dat.add_column(Column(name='veff_max', data=veff_max))

    nsamp=64.0 #over sampling factor
    nbins=2000 #number of volume bins
    nzbins=20 #number of redshift bins (just for plots)
    logvmin=np.log10(1.0e-08) # For the logarithmic binning scheme this is the minimum of the range of Log10(Volume). Volume in (Gpc/h)^3

    for reg in regions:           #Do regions separately (Would have to define volume bins differently to merge)
      print('starting region ',reg)
      regmask=(dat['reg']==reg)#mask to select objects in specified region

      Sel=selection(reg)
      #minimum vmin for the whole selected sample. We just consider volume above this vmin and work in bins of that covering 0 to vmax-vmin
      vmin_sample=Sel['area']*(4.0*np.pi/3.0)*(cosmo.comoving_distance(Sel['zmin']).value)**3 
      vmax_sample=Sel['area']*(4.0*np.pi/3.0)*(cosmo.comoving_distance(Sel['zmax']).value)**3 - vmin_sample
      logvbins=np.linspace(logvmin,np.log10(vmax_sample),nbins) # log volume bin edges
      vbins=10.0**logvbins #edges of bins in the volume coordinate 
      vbins[0]=0.0  # but make very first bin edge exactly zero  
      vbin_width= vbins[1:]  - vbins[:-1] # bin volumes
      vbin_cen=(vbins[:-1]+vbins[1:])/2.0 #volume bin centres
      Delta_vbin=np.ones(vbin_cen.size)  #Overdensity in each volume bin initialised to uniform density
            
      v=np.zeros(dat['Z'].size)          #initialise array for random volume values for each galaxy 
      zbins=np.linspace(Sel['zmin'],Sel['zmax'],nzbins) # zbin edges (used for plots)
      zbin_cen=(zbins[:-1]+zbins[1:])/2.0 #zbin centres

      # For top axis of some plots we set up a grid of redshift tick marks and compute their volume coordinate
      zgrid=np.linspace(0.002,Sel['zmax'],6)
      zgrid_strings = ["%.1f" % number for number in zgrid] #convert to strings rounded to 1 decimal place
      vgrid=Sel['area']*(4.0*np.pi/3.0)*(cosmo.comoving_distance(zgrid).value)**3 - vmin_sample

      #Check that data values are all in expected ranges
      vcheck=  vmax_sample-dat['v'][regmask]
      if (np.min(vcheck)<0): print('ERROR: object with V>V_max')
      vcheck=  vbins[nbins-1]- (dat['v'][regmask]+dat['vmin'][regmask]-vmin_sample)
      if (np.min(vcheck)<0): print('ERROR: object with V+vmin-vmin_sample> largest vbin edge')
      vcheck=  (dat['v'][regmask]+dat['vmin'][regmask]-vmin_sample)
      if (np.min(vcheck)<0): print('ERROR: object with V+vmin<vmin_sample')
      vcheck=  (dat['vmin'][regmask]-vmin_sample)
      if (np.min(vcheck)<0): print('ERROR: object with vmin<vmin_sample')
      zcheck=  (dat['zmin'][regmask]-Sel['zmin'])
      if (np.min(zcheck)<0): print('ERROR: object with zmin<zmin_sample')

      # Plot the number of galaxies per  bin 
      vhist_dat,vbinss=np.histogram((dat['v'][regmask]+dat['vmin'][regmask]-vmin_sample),bins=vbins,weights=dat['WEIGHT'][regmask]) #data histogram in volume coordinate bins
      zhist_dat,zbinss=np.histogram(dat['Z'][regmask],bins=zbins,weights=dat['WEIGHT'][regmask])



      # Set up redshift labels for the top axis
      ax = host_subplot(211)
      ax2 = ax.twin()
      ax2.set_xlabel('$z$')    
      ax2.xaxis.set_label_position('top') 
      ax2.set_xticks(vgrid,labels=zgrid_strings)   
      ax.plot(vbin_cen,np.log10(np.maximum(vhist_dat,1.0e-10)))
      plt.xlim([0,vmax_sample])  
      plt.ylim([0,6.0])
      plt.xlabel('$V /(Mpc/h)^3$')
      plt.ylabel('$log_{10}$ N(V)')
      ax3 = host_subplot(212)
      ax3.plot(zbin_cen,np.log10(np.maximum(zhist_dat,1.0e-10)))
      plt.xlabel('$z$')
      plt.ylabel('$log_{10}$ N(z)')
      plt.show()

      n=np.zeros(dat['Z'].size, dtype=int)        #initialise number of clones of each galaxy to zero
      iv=np.zeros(dat['Z'].size, dtype=int)       
      iv[regmask]=np.digitize(dat['v'][regmask]+dat['vmin'][regmask]-vmin_sample,vbins)-1 #find the volume bin index for each galaxy  

      # Set up redshift labels for the top axis
      ax = host_subplot(111)
      ax2 = ax.twin()
      ax2.set_xlabel('$z$')    
      ax2.xaxis.set_label_position('top') 
      ax2.set_xticks(vgrid,labels=zgrid_strings)  


      for j in range(10): #loop until Deltas converge within Poisson Errors???? -- just loops 6 times for now
        nmean=nsamp*dat['WEIGHT']/dat['Delta'] #mean number of times to redistribute each galaxy

        n[regmask]=np.floor(nmean[regmask]).astype(int) #Round down to the nearest integer
        mask= (np.random.uniform(dat['Delta'].size) < nmean-n) #Choose with probability equal to how much nmean exceeds n those for which n is to be increased by 1
        cmask=mask&regmask #combined mask for region and those needing extra increment 
        n[cmask] += 1 # increment if nmean>n with probability such that <n>=nmean for those in the region
        nmax=n.max() #  the maximum number of times any galaxy will be cloned
        vhist=np.zeros(vhist_dat.size) # initialize vhist histogram of the number of clones in each volume bin
        for i in range(nmax): #loop over the maximum number of times we need to clone any galaxy
            mask= (i<n) #add a clone of all the galaxies that need i or more clones (remember i starts from 0)
            cmask=mask&regmask #combined region and those needing incrementing mask
            nclones=np.count_nonzero(cmask) #this is the total number of galaxies we are cloning on this pass
            # print('pass:',i+1,'of',nmax,' extra clones added:',nclones)
            v[cmask]=dat[cmask]['vmax']*np.random.uniform(size=nclones)# distribute each clone uniformly within its vmax
            hist,vbinss=np.histogram(v[cmask]+dat[cmask]['vmin']-vmin_sample,bins=vbins) #bin the clones in volume coordinate bins
            vhist += hist  # increment the histogram with the clones from this pass
        # Now we have cloned and binned all the galaxies we can compute the overdensity Delta in each volume coordinate bin    
        binmask = (vhist>0)  #mask to avoid divide by zero in any bin with no clones 
        Delta_vbin[binmask]=nsamp*vhist_dat[binmask]/vhist[binmask] #overdensity relative to clones
        print('Volume weight Mean Delta_vbin:',np.sum(Delta_vbin*vbin_width)/np.sum(vbin_width))
        Delta_vbin=Delta_vbin/(np.sum(Delta_vbin*vbin_width)/np.sum(vbin_width)) #Renormalize the mean which is necessary as <nmean>.ne.nsamp as <1/Delta>.ne.1
        print('Normalized Volume weight Mean Delta_vbin:',np.sum(Delta_vbin*vbin_width)/np.sum(vbin_width))
        print('RMS Delta:',np.sqrt(np.mean(((Delta_vbin-np.mean(Delta_vbin))**2))))
        plt.plot(vbin_cen,Delta_vbin,label=str(j))# plot overdensity at this iteration  
        dat['Delta'][regmask]=Delta_vbin[iv[regmask]] # update each galaxy's overdensity (for galaxies in the region)

        #Form the cumulative sum of V_eff up to the bin edges vbin. This is exactly what we need for interpolation.
        veff_bin=np.concatenate((np.array([0.0]),np.cumsum(Delta_vbin*vbin_width)))# cumulative distribution including 0 at first bin edge

        # The first two expressions below would be right if dat['vmin']= vmin_sample but for those with dat['vmin']>vmin_sample we have to make a correction
        veff[regmask]    =np.interp(dat['v'][regmask]   +dat['vmin'][regmask]-vmin_sample, vbins,veff_bin) #For each galaxy interpolate veff from the cumulative binned values
        veff_max[regmask]=np.interp(dat['vmax'][regmask]+dat['vmin'][regmask]-vmin_sample, vbins,veff_bin) #The same for each galaxies Vmax value
        mask=(dat['vmin']>vmin_sample)
        cmask=mask & regmask  #combined mask for those in the region needing their v_eff correcting as dat['vmin']>vmin_sample
        veff[cmask]    =veff[cmask]    - np.interp(dat['vmin'][cmask]-vmin_sample, vbins,veff_bin)
        veff_max[cmask]=veff_max[cmask]- np.interp(dat['vmin'][cmask]-vmin_sample, vbins,veff_bin)  
        #
        plt.scatter(dat['v'][regmask],veff[regmask]-dat['v'][regmask], s=0.25)
        # Go back and do next iteration    
      #
      #Finish off the plot to see how well convergence has worked
      plt.xlim([0,vmax_sample])
      plt.ylabel('$V_{eff} -V$')
      plt.xlabel('$V$')
      ax.legend(loc='upper left')
      plt.show()
    
      # Plot of Delta versus absolute magnitude and Delta versus redshift which should enable understanding of how this LF estimate differs from 1/Vmax
      plt.scatter(dat['ABSMAG_RP1'][regmask],dat['Delta'][regmask],marker='.', linewidths=0, s=0.25, alpha=0.5, label=reg, color=Sel['col'])
      plt.xlabel('$M_r-5 \log h$')
      plt.ylabel('$\Delta$')
      plt.ylim([0.0,6.0])
      plt.legend()
      plt.show()
    
      plt.scatter(dat['Z'][regmask],dat['Delta'][regmask],marker='.', linewidths=0, s=0.25, alpha=0.5, label=reg, color=Sel['col'])
      plt.xlabel('z')
      plt.ylabel('$\Delta$')
      plt.ylim([0.0,6.0])
      plt.legend()
      plt.show()

    # Store the veff and veff_max values that are returned and look at their stats
    dat['veff']=veff
    dat['veff_max']=veff_max
    dat.info('stats')  

    # This was meant to be test of V_eff/V_eff,max is more uniform than V/V_max but to do that it needs to take account the weights but currently does not.
    ratio[regmask]=dat['v'][regmask]/dat['vmax'][regmask]
    plt.hist(ratio[regmask], bins=nbins, label='$V/V_{max}$',histtype='step')
    print('V/V_max uniformity',stats.kstest(ratio[regmask],'uniform'))
    ratio[regmask]=dat['veff'][regmask]/dat['veff_max'][regmask]
    plt.hist(ratio[regmask], bins=nbins, label='$V_{eff}/V_{eff,max}$',histtype='step')
    print('V_eff/V_eff,max uniformity',stats.kstest(ratio[regmask],'uniform'))
    plt.xlabel('$V_{eff}/V_{eff,max}$')
    plt.xlim([0,1.0])
    plt.legend()
    plt.show()
    
    # Plot of Delta versus absolute magnitude and Delta versus redshift which should enable understanding of how this LF estimate differs from 1/Vmax
    plt.scatter(dat['ABSMAG_RP1'][regmask],dat['Delta'][regmask],marker='.', linewidths=0, s=0.25, alpha=0.5, label=reg, color=Sel['col'])
    plt.xlabel('$M_r-5 \log h$')
    plt.ylabel('$\Delta$')
    plt.ylim([0.0,6.0])
    plt.legend()
    plt.show()
    
    plt.scatter(dat['Z'][regmask],dat['Delta'][regmask],marker='.', linewidths=0, s=0.25, alpha=0.5, label=reg, color=Sel['col'])
    plt.xlabel('z')
    plt.ylabel('$\Delta$')
    plt.ylim([0.0,6.0])
    plt.legend()
    plt.show()
    
    
    return 
####################################################################################################################################################################################
# Read John Moustakas's Fast Spec Catalogue
def read_fsf(fpath):
    print('Reading John Moustakas FSF catalogue {}...'.format(fpath))
    fsf = Table.read(fpath)

    mask = (fsf['FLUX_R']>0.0) & (fsf['FLUX_G']>0.0) # remove objects with non-positive flux
    fsf = fsf[mask]    
    #define magnitudes and colours with names compatible with colour_lookup.py
    fsf['REST_GMR_0P1']=fsf['ABSMAG01_SDSS_G']-fsf['ABSMAG01_SDSS_R']
    fsf['RMAG_DRED']  = 22.5 - 2.5*np.log10(fsf['FLUX_R'])
    fsf['GMAG_DRED']  = 22.5 - 2.5*np.log10(fsf['FLUX_G'])
    fsf['gmr_obs']    = fsf['GMAG_DRED'] - fsf['RMAG_DRED']
    #limit redshifts and extreme colours
    mask =  (fsf['Z']>0.01) &  (fsf['Z']<0.6) & (fsf['REST_GMR_0P1']>-0.25) & (fsf['REST_GMR_0P1']<1.5)
    fsf=fsf[mask]
    #define distance modulus and the k-corrections we will fit
    fsf['DISTMOD']    = 25.0+5.0*np.log10(cosmo.luminosity_distance(fsf['Z']).value) 
    fsf['KR_DERIVED']  = fsf['RMAG_DRED'] - fsf['DISTMOD'] - fsf['ABSMAG01_SDSS_R'] 
    fsf['KG_DERIVED']  = fsf['GMAG_DRED'] - fsf['DISTMOD'] - fsf['ABSMAG01_SDSS_G'] 

    return fsf

######################################################################
def remap_rmags(fsf):
  
    
    # Compute rest frame (z=0.1) colours
   
    fsf.info('stats')
    
       
    
    # Compare the N/S distributions of rest frame colour
    mask=(fsf['PHOTSYS']=='S')
    shist,bins=np.histogram(fsf['REST_GMR_0P1'][mask], bins=5000)
    binwidth=bins[1]-bins[0]
    binc=bins[1:]-0.5*binwidth
    shist=shist/(binwidth*mask.sum()) # normalize
    mask=(fsf['PHOTSYS']=='N')
    nhist,bins=np.histogram(fsf['REST_GMR_0P1'][mask], bins=bins)
    nhist=nhist/(binwidth*mask.sum()) # normalize

    plt.plot(binc,shist,label='S')
    plt.plot(binc,nhist,label='N')
    plt.legend()
    plt.ylim([0.0,2.8])
    plt.xlim([-0.25,1.25])
    plt.xlabel('G-R')
    plt.ylabel('P(G-R)')
    plt.show()
    
    # Form cumulative probability distributions 
    scum=np.cumsum(shist)*binwidth
    ncum=np.cumsum(nhist)*binwidth
    plt.plot(binc,scum,label='S')
    plt.plot(binc,ncum,label='N')
    plt.legend()
    plt.ylim([0.0,1.0])
    plt.xlim([-0.25,1.25])
    plt.xlabel('G-R')
    plt.ylabel('P(<(G-R))')
    plt.show()
        
    # For a given N G-R find the S G-R that has the same cumulative probability
    mask=(fsf['PHOTSYS']=='N')
    cprob=np.interp(fsf['REST_GMR_0P1'][mask],binc,ncum) # find cumulative probability from the N distribution
    newcol=np.interp(cprob,ncum,binc) # find colour corresponding to this cumulative probability
    # check this works 
    diff=(newcol-fsf['REST_GMR_0P1'][mask])/binwidth
    print('mean offset in binwidths:',diff.sum()/mask.sum())
    newcol=np.interp(cprob,scum,binc) # find colour corresponding to the S cumulative probability
    
    # Use this colour to define a new r-band magnitude that will produce the required colour
    # and update both the colour and derived k correction to match
    magdiff=fsf['ABSMAG01_SDSS_G'][mask]-newcol -fsf['ABSMAG01_SDSS_R'][mask]
    plt.hist(magdiff,bins=100)
    plt.xlabel('$\Delta M_r$')
    plt.show()
    fsf['ABSMAG01_SDSS_R'][mask]=fsf['ABSMAG01_SDSS_G'][mask]-newcol 
    fsf['REST_GMR_0P1'][mask]=fsf['ABSMAG01_SDSS_G'][mask]-fsf['ABSMAG01_SDSS_R'][mask]
    fsf['KR_DERIVED'][mask]  = fsf['RMAG_DRED'][mask] - fsf['DISTMOD'][mask] - fsf['ABSMAG01_SDSS_R'][mask] 
    
    # Replot overall and in redshift bins
    mask=(fsf['PHOTSYS']=='S')
    shist,bins=np.histogram(fsf['REST_GMR_0P1'][mask], bins=500)
    binwidth=bins[1]-bins[0]
    binc=bins[1:]-0.5*binwidth
    shist=shist/(binwidth*mask.sum()) # normalize
    mask=(fsf['PHOTSYS']=='N')
    nhist,bins=np.histogram(fsf['REST_GMR_0P1'][mask], bins=bins)
    nhist=nhist/(binwidth*mask.sum()) # normalize

    plt.plot(binc,shist,label='S')
    plt.plot(binc,nhist,label='N')
    plt.legend()
    plt.ylim([0.0,3.5])
    plt.xlim([-0.25,1.25])
    plt.xlabel('G-R')
    plt.ylabel('P(G-R)')
    plt.show()

    
    for zbin in np.linspace(0.0,0.5,5):
        
        # Compare the revised N/S distributions of rest frame colour
        mask=(fsf['PHOTSYS']=='S') & (fsf['Z']>zbin) &  (fsf['Z']<zbin+0.1)
        shist,bins=np.histogram(fsf['REST_GMR_0P1'][mask], bins=500)
        binwidth=bins[1]-bins[0]
        binc=bins[1:]-0.5*binwidth
        shist=shist/(binwidth*mask.sum()) # normalize
        mask=(fsf['PHOTSYS']=='N') & (fsf['Z']>zbin) &  (fsf['Z']<zbin+0.1)
        nhist,bins=np.histogram(fsf['REST_GMR_0P1'][mask], bins=bins)
        nhist=nhist/(binwidth*mask.sum()) # normalize

        plt.plot(binc,shist,label='S')
        plt.plot(binc,nhist,label='N')
        plt.legend()
        plt.ylim([0.0,3.5])
        plt.xlim([-0.25,1.25])
        plt.xlabel('G-R')
        plt.ylabel('P(G-R)')
        plt.show()
    
    return fsf
#######################################################################################################   
def  load_full_catalogue_subsets(fullfile):
    
    zmin = 0.0  # Should be the minimum redshift cut used in making the clustering catalogue and should probably be set to exlcude stars i.e. 0.00125 or so
    full = Table.read(fullfile)
    
    # Compute magnitudes and fibre magnitudes and add to the full table
    full.add_column(Column(name='rmag', data=22.5-2.5*np.log10(full['FLUX_R']/full['MW_TRANSMISSION_R']) ))
    full.add_column(Column(name='fib_rmag', data=22.5-2.5*np.log10(full['FIBERFLUX_R']/full['MW_TRANSMISSION_R']) ))
    print('Size of Full catalogue',full['rmag'].size)

    #Throw away objects with fibre mag>23 as they should not be in BGS but this because the photometry changed and we should keep them
    #full = full[(full['fib_rmag']<23.0)] 
    #print('size of Full catalogue if limited to r_fib<23 cut',full['rmag'].size)
    print('Faintest Fibre mag:',np.amax(full['fib_rmag']),'NB BGS was limited to r_fib=23 in an earlier version of the photometry')


    #The observed subset is defined by those reachable by a fibre (ZWARN not Nan) that isn't broken (ZWARN=999999) that achieves a GOODHARDLOC and is predicted to yield decent SNR
    mask = (full['ZWARN']*0 == 0) & (full['ZWARN'] != 999999) & (full['GOODHARDLOC'] == 1)  & (full['TSNR2_BGS']>1000.0)
    #From the observed subset we ideally want to remove all stars but settle for removing just the spectroscopically confirmed stars
    mask = ~mask  | (full['Z_not4clus']<zmin) #true for all the unobserved (~mask) and all the observed that are stars (z<zmin)
    mask = ~mask  # now false for the above, i.e. true for all observed that aren't spectroscopically confirmed to be z<zmin (stars)
    observed=full[mask]
    #The subset of galaxies with good redshift have no ZWARN have large DELTACHI2 and are above zmin (i.e. not stars)
    mask_zgood =(observed['DELTACHI2'] > 40) & (observed['ZWARN'] == 0) & (observed['Z_not4clus']>zmin)
    zgood=observed[mask_zgood]
    
    #zgood.info('stats')
    print('number of objects in the observed subset',  observed['TARGETID'].size)
    print('number of objects in the good redshift subset',zgood['TARGETID'].size)
    #observed.info('stats')
    print('Faintest Observed Fibre mag:',np.amax(observed['fib_rmag']))
    
    
    return zgood, observed
    ###########################################################################################
# extract a subset of the data based on redshift limits and update the zmin,zmax and vmax accordingly
def redhsift_slice(dat,zmin_sub,zmax_sub):
    submask= (dat['Z']>zmin_sub) & (dat['Z']<zmax_sub)
    datsub=dat[submask]
    datsub['zmax'][datsub['zmax']>zmax_sub]=zmax_sub #limit zmax to that of the subsample
    datsub['zmin'][datsub['zmin']<zmin_sub]=zmin_sub #limit zmin to that of the subsample
    # Scale vmax to account for chnages to zmin and zmax
    datsub['vmax']=dat['vmax'][submask]*    ( (cosmo.comoving_distance(datsub['zmax']).value)**3 -(cosmo.comoving_distance(datsub['zmin']).value)**3 ) \
                      /    ( (cosmo.comoving_distance(dat['zmax'][submask]).value)**3 -(cosmo.comoving_distance(dat['zmin'][submask]).value)**3 )
    return datsub
#############################################################################################
def plot_sizes(dat,regions):
    bins = np.arange(0, 40.0, 1.0)
    bin_cen=(bins[:-1] + bins[1:]) / 2.0
    for reg in regions:
        Sel=ca.selection(reg) # define selection parameters for this region
        mask=(dat['reg']==reg)#mask to select objects in specified region        
        count,binz=np.histogram(dat[mask]['SHAPE_R'], bins=bins,  density=True)
        plt.plot(bin_cen, count, label=reg, linewidth=1.0, color=Sel['col'], linestyle=Sel['style'])  
    
    plt.xlabel('SHAPE R')
    plt.ylabel('N')
    plt.xlim([0,20.0])
    plt.ylim([0,0.2])
    plt.legend()
    plt.show()
    return

def plot_depths(dat,regions):
    eps=1.0e-10
    psfmagdepth=22.5-2.5*np.log10(5.0/(eps+np.sqrt(dat['PSFDEPTH_R'])))
    galmagdepth=22.5-2.5*np.log10(5.0/(eps+np.sqrt(dat['GALDEPTH_R'])))
    bins = np.arange(21, 27.0, 0.01)
    bin_cen=(bins[:-1] + bins[1:]) / 2.0
    for reg in regions:
        Sel=ca.selection(reg) # define selection parameters for this region
        mask=(dat['reg']==reg)#mask to select objects in specified region        
        count,binz=np.histogram(psfmagdepth[mask], bins=bins,  density=True)
        plt.plot(bin_cen, count, label='PSF '+reg, linewidth=1.0, color=Sel['col'], linestyle=Sel['style'])  
        count,binz=np.histogram(galmagdepth[mask], bins=bins,  density=True)
        plt.plot(bin_cen, count, label='GAL '+reg, linewidth=1.0, color=Sel['col'], linestyle='dotted') 
    plt.xlabel('r-band PSF DEPTH')
    plt.ylabel('N')
    plt.xlim([22.0,26.0])
    plt.ylim([0,4.0])
    plt.legend()
    plt.show()
    return