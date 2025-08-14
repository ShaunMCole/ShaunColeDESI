#Import required libraries

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import join,Table,Column,vstack,setdiff
from scipy import stats
import catalogue_analysis as ca


#  Assign redshift incompleteness weights from a look-up table that has been pre-computed by redshift_weights.ipynb
#
def assign_redshift_weight(dat,regions,add_sys=False,plot=False):
    print('Assigning redshift weights using pre-computed table.')

    dat['WEIGHT_ZLOOKUP_NGP'] = 1.0   #Default weight to be used outside the grid and in cells in which the completeness is ill-defined
    dat['WEIGHT_ZLOOKUP']     = 1.0
    
    for reg in regions:

        # Read pre-computed redshift weight look-up table
        opath='./data/wz_{}.fits'.format(reg)
        TableH=Table.read(opath) 
        nx=TableH.meta['NX']
        ny=TableH.meta['NY']
        TSNR2_min=TableH.meta['TS_MIN']
        TSNR2_max=TableH.meta['TS_MAX']
        fibmag_bright=TableH.meta['FIB_B']
        fibmag_faint=TableH.meta['FIB_F']
        xbins = np.linspace(TSNR2_min, TSNR2_max, nx+1) #pixel bin edges  for TSNR2_BGS
        ybins = np.linspace(fibmag_bright, fibmag_faint, ny+1)     #pixel bin edges for fib_rmag 
        dxbin=xbins[1]-xbins[0]
        dybin=ybins[1]-ybins[0]
        #print('Grid size for redshift weight lookup table: nx,ny:',nx,ny)
        #convert to a numpy array to allow indexxing
        Harray=np.lib.recfunctions.structured_to_unstructured(TableH.as_array())  

                     
        # For objects in the clustering catalogue within the grid boundaries lookup their completeness and set their new redshift failure weight
        regmask= (dat['reg']==reg)  & (dat['TSNR2_BGS'] > xbins[0]) & (dat['TSNR2_BGS'] < xbins[-1]) # identify objects with the grid in this region
    
        #Use look up to assign the weight
        xvar = 'TSNR2_BGS'
        yvar = 'fib_rmag'
        #Harray=np.asarray(H) # convert to numpy array to enable indexing     
        # use to generate grid indices for the  input data.
        __, __, __, i2 = stats.binned_statistic_2d(dat[xvar][regmask], dat[yvar][regmask], values=dat[xvar][regmask], statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
        j2=i2   #keep the original indices as useful for generating indices of CIC neighbouring cells
        i2=i2-1 #correct for offset in returned bin indices
            
        # check for any points out of range (due to rounding or change in photometry since selection)
        yyvar=dat[yvar][regmask]
        print('number outside boundary forced back to edge',yyvar[i2[1]>ny-1].size,'values follow:',yyvar[i2[1]>ny-1])
        i2[1]=np.minimum(i2[1],ny-1) #force to be within grid points outside due to rounding

        #Normal NGP  look-up
        dat['WEIGHT_ZLOOKUP_NGP'][regmask]=1.0/( Harray[i2[0],i2[1]] )

        #Cloud-in-Cell look-up
        #first compute the difference in the coordinate value of each data point and its nearest grid point in units of the bin spacing
        dx=-0.5-i2[0]+(dat[xvar][regmask]-xbins[0])/dxbin   # these should satisfy -0.5<dx<0.5          
        dy=-0.5-i2[1]+(dat[yvar][regmask]-ybins[0])/dybin   # these should satisfy -0.5<dy<0.5 
        #for each negative value we need to change of j2 to select the correct neighbouring cell (i.e. to left and below rather than above and to the right)
        j2[0][(dx<0)]=j2[0][(dx<0)]-2
        j2[1][(dy<0)]=j2[1][(dy<0)]-2
        #CIC weights  (these add to unity)
        wa=(1.0-np.absolute(dx))*(1.0-np.absolute(dy))
        wb=     np.absolute(dx) *(1.0-np.absolute(dy))
        wc=(1.0-np.absolute(dx))*     np.absolute(dy)
        wd=     np.absolute(dx) *     np.absolute(dy)
        #To avoid out of bounds edge effects replace out of bound cells indices by the NGP cell index
        j0mask = (j2[0]>nx-1) | (j2[0]<0)
        j2[0][j0mask]=i2[0][j0mask]
        j1mask = (j2[1]>ny-1) | (j2[1]<0)
        j2[1][j1mask]=i2[1][j1mask]
        # Form the CIC weighted value and make its inverse the weight
        dat['WEIGHT_ZLOOKUP'][regmask] = 1.0/( wa*Harray[i2[0],i2[1]]+wb*Harray[j2[0],i2[1]]+wc*Harray[i2[0],j2[1]]+wd*Harray[j2[0],j2[1]] )
        print('ZWEIGHT ASSIGNED. to ', np.count_nonzero(regmask))


        #Save the old weights that come with the clustering catalogue
        dat['WEIGHT_OLD'] = dat['WEIGHT']
        #Overwrite the combined weights using the new redshift weight and by default ignoring the systematic weights
        dat['WEIGHT'] = dat['WEIGHT_COMP'] * dat['WEIGHT_ZLOOKUP']
        if add_sys:
           print('WARNING: INCLUDING WEIGHT_SYS!!!')
           dat['WEIGHT'] = dat['WEIGHT_COMP'] * dat['WEIGHT_ZLOOKUP'] * dat['WEIGHT_SYS']






    if (plot):
        #Look at the cumulative weight distribution 
        hist,bins=np.histogram(dat['WEIGHT_ZLOOKUP'], bins=200)
        chist=1.0-np.cumsum(hist)/(dat['WEIGHT_ZLOOKUP'].size)
        binwidth=bins[1]-bins[0]
        binc=bins[1:]-0.5*binwidth
        plt.plot(binc,np.log10(chist+1.0e-20))
        plt.xlim([1.0,7.0])
        plt.ylim([-7.0,0.0])
        plt.xlabel('$w_z$')
        plt.ylabel('$log_{10}\ P(>W_z)$')
        plt.show()

        # Plot the new redshift failure weight against the original one that came with the clustering catalogue
        plt.scatter(dat['WEIGHT_ZLOOKUP'],dat['WEIGHT_ZFAIL'],c='black',s=0.2,linewidth=0,marker='.')
        plt.xlabel('$w_z$')
        plt.ylabel('WEIGHT_ZFAIL')
        plt.show()   

        #check the (Nearest Grid Point) NGP normalization 
        print("NGP weights  (Sum of weights, number observed, difference, fraction)")
        print( np.sum(dat['WEIGHT_ZLOOKUP_NGP']),observed['rmag'].size,np.sum(dat['WEIGHT_ZLOOKUP_NGP'])-observed['rmag'].size,np.sum(dat['WEIGHT_ZLOOKUP_NGP'])/observed['rmag'].size)


        #Check the CIC weight normalization
        print("CIC weights (Sum of weights, number observed, difference, fraction)")
        print( np.sum(dat['WEIGHT_ZLOOKUP']),observed['rmag'].size,np.sum(dat['WEIGHT_ZLOOKUP'])-observed['rmag'].size,np.sum(dat['WEIGHT_ZLOOKUP'])/observed['rmag'].size)

        # Identify stars using a redshift histogram
        zhist,zbins=np.histogram(zgood['Z_not4clus'],bins=50000)
        zbin_width=zbins[1]-zbins[0]
        zbin_cen=zbins[1:]-zbin_width
        plt.plot(zbin_cen,zhist)
        plt.xlim([-0.01,0.6])
        plt.ylim([0.0,5000])
        plt.xlabel('z')
        plt.ylabel('N(z)')
        plt.show()
            

    return 
        

    
# Construct a table of the redshift completeness in bins of fib_rmag and TSNR2_BGS and define a redshift completeness weight by its inverse
#

def construct_weight_table(clus, observed, zgood, plot=True):

   
    
    nx=60
    ny=44
    TSNR2_min=1000
    TSNR2_max=4000
    fibmag_bright=14
    fibmag_faint=23
    xbins = np.linspace(TSNR2_min, TSNR2_max, nx+1) #pixel bin edges  for TSNR2_BGS
    ybins = np.linspace(fibmag_bright, fibmag_faint, ny+1)     #pixel bin edges for fib_rmag 
    dxbin=xbins[1]-xbins[0]
    dybin=ybins[1]-ybins[0]
    print('Number of bins in fibmag and TSNR2_BGS used in weight look-up table. nx,ny:',nx,ny)
   
    #From the observed table add TNSR2_BGS and fib_mag to the clus table
    obs = observed['TARGETID', 'TSNR2_BGS', 'fib_rmag']
    clus = join(clus, obs, keys='TARGETID')

    regions = ['N', 'S'] # Loop over regions so that we deal with each separately

    for reg in regions:

        # select only objects within the region and within the grid boundaries
        zgoodmask = (zgood['PHOTSYS'] == reg)      & (zgood['TSNR2_BGS'] > xbins[0]) & (zgood['TSNR2_BGS'] < xbins[-1])
        obsmask   = (observed['PHOTSYS'] == reg)   & (observed['TSNR2_BGS'] > xbins[0]) & (observed['TSNR2_BGS'] < xbins[-1])
        print('number of observed objects in region and covered by the grid', np.count_nonzero(obsmask))
            
            
    

        #compute grid coordinates for zgood and  observed 
        xg = zgood[zgoodmask]['TSNR2_BGS']
        yg = zgood[zgoodmask]['fib_rmag']
        xall = observed[obsmask]['TSNR2_BGS']
        yall = observed[obsmask]['fib_rmag']

        # In each pixel count the number of galaxies with redshift and the number observed
        Hg, xedges_g, yedges_g, binnumber_g = stats.binned_statistic_2d(xg, yg, None, 'count', bins=[xbins, ybins], expand_binnumbers=True)
        Hall, xedges_all, yedges_all, binnumber_all = stats.binned_statistic_2d(xall, yall, None, 'count', bins=[xbins, ybins], expand_binnumbers=True)
        # Avoiding divide by zero, set default completeness and compute completeness in the occupied pixels
        H = Hall*0.0+1.0 # pre-set all H values to a default of 1.0
        nonzero_mask = (Hall >0) & (Hg >0)
        missed_mask =(Hall >0) & (Hg == 0)  # these are galaxies that don't have their weight passed to others as observed galaxies in the pixel but none with redshifts
        print(reg,': missed and unrepresented objects',np.sum(Hall[missed_mask]))        
        H[nonzero_mask] = (Hg[nonzero_mask]/Hall[nonzero_mask]) #completeness in this pixel.


            

        # Make a plot of the completeness look-up table with the catalogue points overlaid
        if plot:
                print('The following plots are the raw look-up table and as seen with CIC interpolation:')
                XXg, YYg = np.meshgrid(xbins, ybins)
                fig = plt.figure(figsize = (13,7))
                ax1=plt.subplot(111)
                plot1 = ax1.pcolormesh(XXg,YYg,H.T, vmin=0.2)
                cbar = plt.colorbar(plot1,ax=ax1, pad = .015, aspect=10)
                plt.scatter(zgood[zgoodmask]['TSNR2_BGS'],zgood[zgoodmask]['fib_rmag'], marker='.',linewidth=0,s=0.1,alpha=0.1,c='red')
                plt.xlabel('TSNR2_BGS')
                plt.ylabel('r_fibre')
                plt.xlim([1000.0,4000.0])
                plt.ylim([14.0,23.0])
                plt.show()
                
                
                # An attempt to visualise CiC lookup on a finer grid
                nm=20
                xfbins = np.linspace(1000, 4000, nm*nx+1) #pixel bin edges  for TSNR2_BGS
                yfbins = np.linspace(14, 23, nm*ny+1)     #pixel bin edges for fib_rmag 
                XXg, YYg = np.meshgrid(xfbins, yfbins)
                Harray=np.asarray(H) # convert to numpy array to enable indexing     
                # use to generate grid indices for the  input data.
                __, __, __, i2 = stats.binned_statistic_2d(XXg.flatten(), YYg.flatten(), values=XXg.flatten(), statistic='median', bins = [xbins, ybins], expand_binnumbers=True)
                j2=i2   #keep the original indices as useful for generating indices of CIC neighbouring cells
                i2=i2-1 #correct for offset in returned bin indices
                #Cloud-in-Cell look-up
                #first compute the difference in the coordinate value of each data point and its nearest grid point in units of the bin spacing
                dx=-0.5-i2[0]+(XXg.flatten()-xbins[0])/dxbin   # these should satisfy -0.5<dx<0.5          
                dy=-0.5-i2[1]+(YYg.flatten()-ybins[0])/dybin   # these should satisfy -0.5<dy<0.5
                #for each negative value we need to change of j2 to select the correct neighbouring cell (i.e. to left and below rather than above and to the right)
                j2[0][(dx<0)]=j2[0][(dx<0)]-2
                j2[1][(dy<0)]=j2[1][(dy<0)]-2
                #CIC weights  (these add to unity)
                wa=(1.0-np.absolute(dx))*(1.0-np.absolute(dy))
                wb=     np.absolute(dx) *(1.0-np.absolute(dy))
                wc=(1.0-np.absolute(dx))*     np.absolute(dy)
                wd=     np.absolute(dx) *     np.absolute(dy)
                #To avoid out of bounds edge effects replace out of bound cells indices by the NGP cell index
                j0mask = (j2[0]>nx-1) | (j2[0]<0)
                j2[0][j0mask]=i2[0][j0mask]
                j1mask = (j2[1]>ny-1) | (j2[1]<0)
                j2[1][j1mask]=i2[1][j1mask]
                # Form the CIC weighted value and make its inverse the weight
                FH =  wa*Harray[i2[0],i2[1]]+wb*Harray[j2[0],i2[1]]+wc*Harray[i2[0],j2[1]]+wd*Harray[j2[0],j2[1]] 
                FH=FH.reshape(nm*ny+1,nm*nx+1)
                fig = plt.figure(figsize = (13,7))
                ax1=plt.subplot(111)
                plot1 = ax1.pcolormesh(XXg,YYg,FH, vmin=0.2)
                cbar = plt.colorbar(plot1,ax=ax1, pad = .015, aspect=10)
                plt.scatter(zgood[zgoodmask]['TSNR2_BGS'],zgood[zgoodmask]['fib_rmag'], marker='.',linewidth=0,s=0.1,alpha=0.1,c='red')
                plt.xlabel('TSNR2_BGS')
                plt.ylabel('r_fibre')
                plt.xlim([1000.0,4000.0])
                plt.ylim([14.0,23.0])
                plt.show()

        # save the look-up table for future use
        TableH=Table(np.asarray(H)) 
        TableH.meta['TS_MIN']=TSNR2_min
        TableH.meta['TS_MAX']=TSNR2_max
        TableH.meta['FIB_B']=fibmag_bright
        TableH.meta['FIB_F']=fibmag_faint
        TableH.meta['NX']=nx
        TableH.meta['NY']=ny
        opath='./data/wz_{}.fits'.format(reg)
        TableH.write(opath,overwrite=True) 
        print('LOOK-UP TABLE GENERATED AND SAVED.')    
            
    return clus

