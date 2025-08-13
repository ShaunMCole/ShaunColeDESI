import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from astropy.table import Table, join
import catalogue_analysis as ca


def pc10(x):  #returns 10th percentile (for use in binned_statistics)
    pc10=np.percentile(x,10.0)
    return pc10

def pc90(x): #returns 90th percentile (for use in binned_statistics)
    pc90=np.percentile(x,90.0)
    return pc90


# Builds lookup table 
def construct_colour_table(fsf, reg, plot=True):

    # Parameters of the redshift-colour grid will be saved alongside the table
    dcolbin=0.025    
    dzbin=0.0025
    zmin=0.01
    zmax=0.6001
    colmin=-1.0
    colmax=4.001
    xbins = np.arange(zmin, zmax, dzbin)
    ybins = np.arange(colmin, colmax, dcolbin)
    ny=np.size(ybins)-2
    nx=np.size(xbins)-2
    #print('Grid size for colour lookup table: nx,ny:',nx,ny)
    
    
    print('GENERATING NEW LOOKUP TABLE.')
    # Make a copy of the orginal FSF magnitudes to use in comparison plots
    fsf['REST_GMR_0P1_original'] = fsf['REST_GMR_0P1'].copy()
        
    
    rmask = (fsf['PHOTSYS'] == reg)   #select just N or S data as each has to be modelled separately
       
    # extract the data to x,y,z vectors for compatibility with the binned_statistic_2d() function
    x = fsf['Z'][rmask]
    y = fsf['gmr_obs'][rmask]
    z = fsf['REST_GMR_0P1_original'][rmask]
    # Determine the median rest frame colour (z) and object ocunt in each bin of redshift (x)  and observer frame colour (y)
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z, statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
    count, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z, statistic='count', bins = [xbins, ybins], expand_binnumbers=True)
    XX, YY = np.meshgrid(xedges, yedges)
   
    # Replace noisy and empty pixel values by one of two extrapolation methods
    sam = False
    if (sam):
          # replace noisy and empty pixel values by neighouring values at the same redshift
          for idx in range(len(H)):
              mask = np.isnan(H[idx])
              try:
                  H[idx][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), H[idx][~mask])
              except:
                  pass # i.e: if whole column is empty
                # save the look-up table for future use
                    
    shaun= True        
    if (shaun):    
          colmid=0.8   # A mid range colour in the well populated band that needs no extrapolation 
          jmid=(colmid-ybins[0])/dcolbin # corresponding bin index

          #Extrapolate to bins with zero or low count be replacing with the value from the nearest bin in colour at the same redshift with count above a threshold
          for j in range(len(yedges)-1):  # loop over the bins
            for i in range(len(xedges)-1):                
                if (count[i][j]<2.5):  # For any bin with a count of 2 or less
                    if (j>jmid):       # If redder than colmid replace with the value from the reddest bin with count greater than 3
                        j1=j
                        while ((count[i][j1]<3.5) & (j1>jmid)):
                           j1=j1-1
                           H[i][j]=H[i][j1]    
                    if (j<=jmid):       # If bluer than colmid replace with the value from the bluest bin with count greater than 3
                        j1=j
                        j1=j
                        while ((count[i][j1]<3.5) & (j1<=jmid)):
                           j1=j1+1
                           H[i][j]=H[i][j1] 
        
    # Convert the Table in an astro-py table and store along with meta data that determines grid spec
    TableH=Table(np.asarray(H)) 
    TableH.meta['DZBIN']=dzbin
    TableH.meta['ZMIN']=zmin
    TableH.meta['ZMAX']=zmax
    TableH.meta['DCOLBIN']=dcolbin
    TableH.meta['COLMIN']=colmin
    TableH.meta['COLMAX']=colmax
    TableH.meta['NX']=nx
    TableH.meta['NY']=ny
    opath='./data/ColourLookupTable_{}.fits'.format(reg)
    TableH.write(opath,overwrite=True) 
    print('LOOK-UP TABLE GENERATED AND SAVED.')
    # now create corresponding k-corrections in bins of the assigned colour in the region we have already restricted to
    all_bins, all_medians = gen_kcorr(fsf, reg, colval='REST_GMR_0P1', nbins=10, write=True, fill_between=True)
    # Use the table to assign new rest frame colours to the FSF catalogue galaxies
    regmask=(fsf['PHOTSYS']==reg)
    print("Reloading look-up table to apply to the FSF catalogue")
    colour_table_lookup(fsf, regmask, reg, replot=False, fresh=False)  
    
    # Test plots if requested    
    if plot:  # Compare colours from the look-up table with John's original colours
           print('Comparing the distributions of newly assigned colours with those of the originals')
           for zmin in np.linspace(0.0,0.5,num=5): #For redshift bins plot histograms of the restframe colour distributions
             zmask= (fsf['Z']> zmin) &(fsf['Z']< zmin+0.125)
             mask=zmask & rmask
             jhist,bins=np.histogram(fsf['REST_GMR_0P1_original'][mask],bins=40,density=True)        
             lhist,bins=np.histogram(fsf['REST_GMR_0P1'][mask],bins=bins,density=True)
             binc= (bins[:-1]+bins[1:])/2.0   
             plt.plot(binc,jhist,'k-',label='John Moustakas')
             plt.plot(binc,lhist,'r-',label='Look-up Table')
             plt.xlabel(r'$g-r$')
             plt.ylabel(r'N')
             plt.legend()
             plt.title('Colour distribution '+str(zmin)+'<z<'+str(zmin+0.125))
             plt.show()
           for zmin in np.linspace(0.05,0.55,num=5): #For redshift slices make colour magnitude plots
             zmask= (fsf['Z']> zmin) &(fsf['Z']< zmin+0.01)
             mask=zmask & rmask
             plt.scatter(fsf['ABSMAG01_SDSS_R'][mask],fsf['REST_GMR_0P1_original'][mask], marker='.', linewidths=0, alpha=0.5, s=0.25, color='black')
             plt.scatter(fsf['ABSMAG01_SDSS_R'][mask],fsf['REST_GMR_0P1'][mask], marker='.', linewidths=0, alpha=0.5, s=0.25, color='red')
             plt.ylabel(r'$g-r$')
             plt.title('Colour distribution '+str(zmin)+'<z<'+str(zmin+0.01))
             plt.xlim([-24,-19])
             plt.show()


     
        
    if plot == True:  #If requested make some plots to verify the k-corrections
              # k-correction difference plots.
              kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg)

              prop_cycle = plt.rcParams['axes.prop_cycle']
              colors = prop_cycle.by_key()['color']
                
                
              k_corr_file = './data/jmext_kcorr_{}_rband_z01.dat'.format(reg)
              

              col_min, col_max, A, B, C, D, E, F, G, col_med = np.loadtxt(k_corr_file, unpack=True)
                
                
              #  Plot median k corrections and the polynomial fits to them
              print('plots of the run of median k-corrections, the polynomial fits and the difference')
              fig,ax = plt.subplots(2,1, figsize=(5,10))
              prop_cycle = plt.rcParams['axes.prop_cycle']
              colours = prop_cycle.by_key()['color']
            
              for idx in range(len(all_bins)):
                    z       = all_bins[idx]
                    medians = all_medians[idx]
                    rest_GMR     = col_med[idx]
                    GMR          = rest_GMR * np.ones_like(z)
                    k            = kcorr_r.k(z, GMR)

                    ax[0].plot(z, medians, color=colours[idx], label='{:.5f}'.format(rest_GMR))
                    ax[0].plot(z, k, color=colours[idx])

                    ax[1].plot(z, medians-k, color=colours[idx])
                    ax[1].axhline(0.0, ls='--', lw=0.5, color='black')

              #ax[0].legend()
              ax[1].set_xlabel('Z')
              ax[0].set_ylabel(r'$k_r$')
              ax[1].set_ylabel(r'$k_{r,med} - k_{r, curve}$')
              plt.show()

             
          
              #Compare the absolute magnitudes computed using the polynomial k-corrections to John Moustakas' orginal ones  
              fsf['ABSMAG_SDSS_R_SAM'] = -99.9 # create an array to store the assigned absolute magnitudes
              rmask = (fsf['PHOTSYS'] == reg)
              Qzero = 0.0 # no evolution correction to be consistent with John
              fsf['ABSMAG_SDSS_R_SAM'][rmask]=ca.ABSMAG(fsf['RMAG_DRED'][rmask],fsf['Z'][rmask],fsf['REST_GMR_0P1'][rmask],kcorr_r,Qzero)

              diff = fsf['ABSMAG_SDSS_R_SAM'][rmask]-fsf['ABSMAG01_SDSS_R'][rmask]
              bin_medians, bin_edges, binnumber = stats.binned_statistic(fsf['Z'][rmask], diff, statistic='median', bins=25)
              bin_10pc, bin_edges, binnumber = stats.binned_statistic(fsf['Z'][rmask], diff, statistic=pc10, bins=25)
              bin_90pc, bin_edges, binnumber = stats.binned_statistic(fsf['Z'][rmask], diff, statistic=pc90, bins=25)  
            
              plt.scatter(fsf['Z'][rmask],diff, marker='.', color='red', linewidths=0,s=0.2,alpha=0.4)
              centres = (bin_edges[:-1]+bin_edges[1:])/2
              plt.plot(centres, bin_medians, label='median')
              plt.plot(centres, bin_10pc, label='10%')
              plt.plot(centres, bin_90pc, label='90%')
              plt.ylim(-0.15, 0.15)
              plt.axhline(0.0, ls='--', color='black')
              plt.xlabel('Z')
              plt.ylabel('SM-JM Mr')
              plt.legend(loc='upper left')
              plt.show()

    return
        



def colour_table_lookup(dat, regmask, reg, replot=True, fresh=False): 
        print('LOADING IN LOOKUP TABLE')
        opath='./data/ColourLookupTable_{}.fits'.format(reg)
        TableH=Table.read(opath) 
        # Parameters of the redshift-colour grid
        dzbin=TableH.meta['DZBIN']
        zmin=TableH.meta['ZMIN']
        zmax=TableH.meta['ZMAX']
        dcolbin=TableH.meta['DCOLBIN']
        colmin=TableH.meta['COLMIN']
        colmax=TableH.meta['COLMAX']
        xbins = np.arange(zmin, zmax, dzbin)
        ybins = np.arange(colmin, colmax, dcolbin)
        ny=np.size(ybins)-2
        nx=np.size(xbins)-2
        #print('Grid size for colour lookup table: nx,ny:',nx,ny)
        assert TableH.meta['NX'] == nx, f"Grid dimension nx does not match: {TableH.meta['NX']} != {nx}"
        assert TableH.meta['NY'] == ny, f"Grid dimension ny does not match: {TableH.meta['NY']} != {ny}"
        #Convert the Table to a numpy array for compatibility with functions used below
        Harray=np.lib.recfunctions.structured_to_unstructured(TableH.as_array())  
    
     
        if replot: # make a plot of the look-up table
            XX, YY = np.meshgrid(xbins, ybins)
            fig   = plt.figure(figsize = (13,7))
            ax1   = plt.subplot(111)
            plot1 = ax1.pcolormesh(XX,YY,Harray.T)
            label = r'$(g-r)_{0, med}$'
            cbar = plt.colorbar(plot1, ax=ax1, pad = .015, aspect=10, label=label)
            plt.ylim(-0.2, 2)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$g-r$')
            plt.title('Lookup Table ({})'.format(reg))
            plt.show() 


       
        # use to generate grid indices for the  input data.
        __, __, __, i2 = stats.binned_statistic_2d(dat['Z'][regmask], dat['gmr_obs'][regmask], values=dat['Z'][regmask], statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
        j2=i2   #keep the original indices as useful for generating indices of CIC neighbouring cells
        i2=i2-1 #correct for offset in returned bin indices
        i2[0]=np.clip(i2[0],a_min=0,a_max=nx) #avoid rounding errors causing out of bounds index
        i2[1]=np.clip(i2[1],a_min=0,a_max=ny)
       
 
        #Cloud-in-Cell look-up
        #first compute the difference in the coordinate value of each data point and its nearest grid point in units of the bin spacing
        dx=-0.5-i2[0]+(dat['Z'][regmask]-xbins[0])/dzbin           # these should satisfy -0.5<dx<0.5          
        dy=-0.5-i2[1]+(dat['gmr_obs'][regmask]-ybins[0])/dcolbin   # these should satisfy -0.5<dy<0.5 
        #for each negative value we need to change of j2 to select the correct neighbouring cell 
        j2[0][(dx<0)]=j2[0][(dx<0)]-2
        j2[1][(dy<0)]=j2[1][(dy<0)]-2
        #CIC weights  (these add to unity)
        wa=(1.0-np.absolute(dx))*(1.0-np.absolute(dy))
        wb=     np.absolute(dx) *(1.0-np.absolute(dy))
        wc=(1.0-np.absolute(dx))*     np.absolute(dy)
        wd=     np.absolute(dx) *     np.absolute(dy)
        #To avoid out of bounds edge effects replace out of bound cells indices by the NGP cell index
        j0mask = (j2[0]>nx) | (j2[0]<0)
        j2[0][j0mask]=i2[0][j0mask]
        j1mask = (j2[1]>ny) | (j2[1]<0)
        j2[1][j1mask]=i2[1][j1mask]
        # Form the CIC weighted value
        dat['REST_GMR_0P1'][regmask] = wa*Harray[i2[0],i2[1]]+wb*Harray[j2[0],i2[1]]+wc*Harray[i2[0],j2[1]]+wd*Harray[j2[0],j2[1]]
        print('REST-FRAME COLOURS ASSIGNED.')
        return
    
def kcorr_table(mins, maxs, polys, medians, split_num, opath, print_table=False):
    
    #print('TABLE OPATH:', opath)
    
    header = "# 'gmr_min', 'gmr_max', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'gmr_med'\n"
    #print(header)
    
    if opath:
        f = open(opath, "w")
        f.writelines([header])
    
    results = []
    for idx in range(split_num):
        result = "{} {} {} {} {} {} {} {} {} {}\n".format(mins[idx], maxs[idx], polys[idx][0], polys[idx][1], polys[idx][2], polys[idx][3], polys[idx][4], polys[idx][5], polys[idx][6], medians[idx])
        if print_table:
            print(result)
        
        results.append(result)
        
        if opath:
            f.writelines([result])

    if opath:
        #result.write(opath, format='fits', overwrite=True)
        f.close()
        

def func(z, a0, a1, a2, a3, a4, a5, a6):
    zref = 0.1
    x = z-zref
    return a0*x**6 + a1*x**5 + a2*x**4 + a3*x**3 + a4*x**2 + a5*x**1 + a6



def gen_kcorr(fsf, regions, colval='REST_GMR_0P1', nbins=10, write=False, rolling=False, adjuster=False, fill_between=False):

    from scipy import interpolate
    from scipy import stats
    from scipy.optimize import curve_fit
    from scipy.interpolate import splev, splrep, UnivariateSpline
    import numpy as np

    dat_everything = fsf
    bands = ['R']

    if rolling:
        nbins += 2

    colours = plt.cm.jet(np.linspace(0,1,nbins+2))

    for photsys in regions:
        for band in bands:

            photmask = (dat_everything['PHOTSYS'] == photsys)
            dat_all = dat_everything[photmask]
            # print('PHOTSYS={}, BAND={}, LENGTH={}'.format(photsys, band, len(dat_all)))

            percentiles = np.arange(0, 1.01, 1/nbins) 
            bin_edges = stats.mstats.mquantiles(dat_all[colval], percentiles)
            dat_all['COLOUR_BIN'] = np.digitize(dat_all[colval], bin_edges)

            if rolling:
                # rolling colour bins - redefine digitisation for finer initial binning.
                bin_edges = np.linspace(0, 1.25, nbins)
                dat_all['COLOUR_BIN'] = np.digitize(dat_all[colval], bin_edges)

            col_var = 'COLOUR_BIN'
            col_min_all = []
            col_max_all = []

            mins = []
            maxs = []
            polys = []
            medians = []
            all_bins    = []
            all_medians = []
            
            start, end = 1, nbins

            if rolling:
                start = 1
                end = nbins - 1

            for idx in np.arange(start,end):
                mask = (dat_all[col_var] == idx)

                # extend the masks to make 'rolling' fits
                if rolling:
                    mask = (dat_all[col_var] == idx-1) | (dat_all[col_var] == idx) | (dat_all[col_var] == idx+1)

                dat = dat_all[mask]
                #print(idx, len(dat))

                col_min = min(dat[colval])
                col_max = max(dat[colval])
                col_med = np.median(dat[colval])

                yvar = 'K{}_DERIVED'.format(band.upper())

                x = dat['Z']
                y = dat[yvar]
                x = x[~np.isnan(y)]
                y = y[~np.isnan(y)]
                x = x[np.isfinite(y)]
                y = y[np.isfinite(y)]

                # NEW MEDIAN test
                total_bins = 100
                bin_medians, bin_edges, binnumber = stats.binned_statistic(x, y,statistic='median', bins=total_bins)
                bin_width = (bin_edges[1] - bin_edges[0])
                bin_centres = bin_edges[1:] - bin_width/2
                bins = bin_centres
                running_median = bin_medians

                p0 = [1.5,  0.6, -1.15,  0.1, -0.16]
                popt, pcov = curve_fit(func, bins, running_median, maxfev=5000)

                mins.append(col_min)
                maxs.append(col_max)
                polys.append(popt)
                medians.append(col_med)
                all_bins.append(bins)
                all_medians.append(running_median)
                
                #plt.scatter(dat['Z'], dat[yvar], s=0.25, label=r'${:.2f}<(g-r)_0<{:.2f}, n={}$'.format(col_min, col_max, len(dat)))
                #label='median (idx={})'.format(idx)
                label= None
                #plt.plot(bins,running_median, marker=None,fillstyle='none',markersize=5,alpha=0.5,color=colours[idx],label=label)

                #label='fit: a0=%5.3f, a1=%5.3f, a2=%5.3f, a3=%5.3f, a4=%5.3f' % tuple(popt)
                label = None
                #plt.plot(x, func(x, *popt), color=colours[idx], ls='--',label=label)

                if fill_between:
                    if idx % 2 != 0:
                        try:
                            plt.fill_between(bins,plow,phigh,color='blue',alpha=0.25)            
                        except:
                            pass

            #plt.xlabel('z')
            #plt.ylabel('K(z)')
            #plt.ylim(-1, 1)
            #plt.title('FSF (main, v2.0), PHOTSYS={}, band={}'.format(photsys, band))
            #plt.show()

            opath = './data/jmext_kcorr_{}_{}band_z01.dat'.format(photsys.upper(), band.lower())

            #opath = '/global/u2/s/smcole/DESI/NvsS/data/jmext_kcorr_{}_{}band_z01.dat'.format(photsys.upper(), band.lower())

            
            
            #TODO: Check this.
            split_num = end-start

            if rolling:
                split_num = nbins - 2

            if write:
                print('Writing k-correction polynomials to {}'.format(opath))
                kcorr_table(mins, maxs, polys, medians, split_num, opath)

    bins = np.arange(-0.5, 1.5, 0.01)
    plt.hist(dat_all[colval], bins=bins)

    for col_split in maxs[0:-1]:
        plt.axvline(col_split, ls='--', lw=0.25, color='r')

    plt.xlabel(r'$(g-r)_0$')
    plt.ylabel('N')
    plt.title('FSF (main, v2.0), PHOTSYS={}, band={}'.format(photsys, band))
    #plt.show()
    
    return all_bins, all_medians 

    import numpy as np
import matplotlib.pyplot as plt
from   scipy.interpolate import interp1d
from   pkg_resources     import resource_filename




class DESI_KCorrection(object):
    def __init__(self, band, kind="cubic", file='jmext', photsys=None):
        """
        Colour-dependent polynomial fit to the FSF DESI K-corrections, 
        used to convert between SDSS r-band Petrosian apparent magnitudes, and rest 
        frame absolute manigutues at z_ref = 0.1
        
        Args:
            k_corr_file: file of polynomial coefficients for each colour bin
            z0: reference redshift. Default value is z0=0.1
            kind: type of interpolation between colour bins,
                  e.g. "linear", "cubic". Default is "cubic"
        """
        
        import os
        os.environ['CODE_ROOT'] = '/global/u2/s/smcole/DESI/NvsS'
        raw_dir = os.environ['CODE_ROOT'] + '/data/'        
        #print(os.environ['CODE_ROOT'])
        
        # check pointing to right directory.
        # print('FILE:', file)
        # print('PHOTSYS:', photsys)
        
        if file == 'ajs':
            k_corr_file = raw_dir + '/ajs_kcorr_{}band_z01.dat'.format(band.lower())
                 
        elif file == 'jmext':
            k_corr_file = raw_dir + '/jmext_kcorr_{}_{}band_z01.dat'.format(photsys.upper(), band.lower())
            
        elif file == 'jmextcol':
            # WARNING: These are the colour polynomials.
            # TODO: separate this to avoid ambiguity.
            k_corr_file = raw_dir + '/jmextcol_kcorr_{}_{}band_z01.dat'.format(photsys.upper(), band.lower())
            
            
        else:
            print('FILE NOT SUPPORTED.')
            
        print('Reading k-correction polynomials from ',k_corr_file)            
        # read file of parameters of polynomial fit to k-correction
        # polynomial k-correction is of the form
        # A*(z-z0)^6 + B*(z-z0)^5 + C*(z-z0)^4 + D*(z-z0)^3 + ... + G
        col_min, col_max, A, B, C, D, E, F, G, col_med = \
            np.loadtxt(k_corr_file, unpack=True)
    
        self.z0 = 0.1            # reference redshift

        self.nbins = len(col_min) # number of colour bins in file
        self.colour_min = np.min(col_med)
        self.colour_max = np.max(col_med)
        self.colour_med = col_med

        # functions for interpolating polynomial coefficients in rest-frame color.
        self.__A_interpolator = self.__initialize_parameter_interpolator(A, col_med, kind=kind)
        self.__B_interpolator = self.__initialize_parameter_interpolator(B, col_med, kind=kind)
        self.__C_interpolator = self.__initialize_parameter_interpolator(C, col_med, kind=kind)
        self.__D_interpolator = self.__initialize_parameter_interpolator(D, col_med, kind=kind)
        self.__E_interpolator = self.__initialize_parameter_interpolator(E, col_med, kind=kind)
        self.__F_interpolator = self.__initialize_parameter_interpolator(F, col_med, kind=kind)
        self.__G_interpolator = self.__initialize_parameter_interpolator(G, col_med, kind=kind)

        # Linear extrapolation for z > 0.5
        self.__X_interpolator = lambda x: None
        self.__Y_interpolator = lambda x: None
        self.__X_interpolator, self.__Y_interpolator = self.__initialize_line_interpolators() 
   
    def __initialize_parameter_interpolator(self, parameter, median_colour, kind="linear"):
        # returns function for interpolating polynomial coefficients, as a function of colour
        return interp1d(median_colour, parameter, kind=kind, fill_value="extrapolate")
    
    def __initialize_line_interpolators(self):
        # linear coefficients for z>0.5
        X = np.zeros(self.nbins)
        Y = np.zeros(self.nbins)
        
        # find X, Y at each colour
        redshift = np.array([0.58,0.6])
        arr_ones = np.ones(len(redshift))
        for i in range(self.nbins):
            k = self.k(redshift, arr_ones*self.colour_med[i])
            X[i] = (k[1]-k[0]) / (redshift[1]-redshift[0])
            Y[i] = k[0] - X[i]*redshift[0]
        
        X_interpolator = interp1d(self.colour_med, X, kind='linear', fill_value="extrapolate")
        Y_interpolator = interp1d(self.colour_med, Y, kind='linear', fill_value="extrapolate")
        
        return X_interpolator, Y_interpolator

    def __A(self, colour):
        # coefficient of the z**6 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__A_interpolator(colour_clipped)

    def __B(self, colour):
        # coefficient of the z**5 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__B_interpolator(colour_clipped)

    def __C(self, colour):
        # coefficient of the z**4 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__C_interpolator(colour_clipped)

    def __D(self, colour):
        # coefficient of the z**3 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__D_interpolator(colour_clipped)
    
    def __E(self, colour):
        # coefficient of the z**2 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__E_interpolator(colour_clipped)
    
    def __F(self, colour):
        # coefficient of the z**1 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__F_interpolator(colour_clipped)
    
    def __G(self, colour):
        # coefficient of the z**0 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__G_interpolator(colour_clipped)

    def __X(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__X_interpolator(colour_clipped)

    def __Y(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__Y_interpolator(colour_clipped)



    def k(self, redshift, restframe_colour, median=False):
        """
        Polynomial fit to the DESI
        K-correction for z<0.6
        The K-correction is extrapolated linearly for z>0.6

        Args:
            redshift: array of redshifts
            colour:   array of ^0.1(g-r) colour
        Returns:
            array of K-corrections
        """
        K   = np.zeros(len(redshift))
        idx = redshift <= 0.6
        
        if median:
            restframe_colour = np.copy(restframe_colour)
            
            # Fig. 13 of https://arxiv.org/pdf/1701.06581.pdf            
            restframe_colour = 0.603 * np.ones_like(restframe_colour)

        K[idx] = self.__A(restframe_colour[idx])*(redshift[idx]-self.z0)**6 + \
                 self.__B(restframe_colour[idx])*(redshift[idx]-self.z0)**5 + \
                 self.__C(restframe_colour[idx])*(redshift[idx]-self.z0)**4 + \
                 self.__D(restframe_colour[idx])*(redshift[idx]-self.z0)**3 + \
                 self.__E(restframe_colour[idx])*(redshift[idx]-self.z0)**2 + \
                 self.__F(restframe_colour[idx])*(redshift[idx]-self.z0)**1 + \
                 self.__G(restframe_colour[idx])

        idx = redshift > 0.6
        
        K[idx] = self.__X(restframe_colour[idx])*redshift[idx] + self.__Y(restframe_colour[idx])
        
        return  K    

    