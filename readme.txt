Shaun Cole's collection of code for DESI Analysis

This is backed up in the following GitHub Repository: git@github.com:ShaunMCole/ShaunColeDESI.git

Collections of Python functions:
   catalogue_analysis.py   -- wide collection of python code for analysing galaxy catalogues including estimating luminosity functions
   rootfinders.py          -- rootfinder to find the roots of a vector of equations such as finding zmax for a catalogue of galaxies  
   kcorrections.py         -- code for setting up and applying k-corrections
   redshift_weights.py     -- code for computing weights to correct for how redshift completeness depends on fibre mag and target SNR

iPython notebooks:
    Augment_BGS_cat.ipynb  -- take the standard BGS clustering catalogue and augment with extra data such as Vmax and jackknife region
    CompareCats.ipynb      -- graphcial comparison of two catalogues matched by targetid to see what has changed
    Pertrosian.ipynb       -- contains some useful cocde for computing model petrosian magnitudes
    Galactocentric.ipynb   -- contains some code useful for converting between helio-centric and galacto-centric redshifts
    Y3.ipyn                -- Standard analysis of the augmented Y3 catalogue including various global LF estimates   
    Slices_vollim.ipynb    -- Variant of the above in which data is split into redshift slices, LFs estimated and stored in named file
    LFPlotting.ipynb       -- Reads the LFs tabulated for each redshifts slice, manipulates and plots   

Data files:
  Primary:  (can be copied from a DESI repository or elsewhere)
     ./data/fastspec-iron-main-bright.fits -- John Moustakas FastSpecFit catalogue
  Secondary:   (can be created by Augment_BGS_cat.ipynb)
     ./data/colourLookupTable_N.fits       -- restframe colour look up table created from fastspec catalogue for North 
     ./data/colourLookupTable_S.fits       -- restframe colour look up table created from fastspec catalogue for South 
     ./data/jmext_kcorr_N_rband_z01.dat    -- tabulated kcorrection polynomials for North
     ./data/jmext_kcorr_S_rband_z01.dat    -- tabulated kcorrection polynomials for South
     ./data/wz_N.fits                      -- tabulated redshift completeness weights for North
     ./data/wz_S.fits                      -- tbaulated redshift completeness weights for South  
   Tertiary
      LF_Sept2025.fits                     -- LF estimates in redshift slices 

Functions in catalogue_analysis.py
    selection(reg)   :   sets up the selection cuts for each region
    Y3load_catalogues(fpath) : Load the Y3 LSS catalogue
    load_catalogues(fpathN,fpathS) : Load earlier LSS catalogues that were split N and S
    load_catalogue(fpath) : load older LSS catalogue
    redshiftslices(dat,zbin_edges,regions,plotfrac=0.2) : Flag galaxies by redshift slice, flag as whether in volume limited subset etc
    solve_jackknife_nonsq(data, ndiv_ra=4, ndiv_dec=5, offset=275) : Define a set of jackknife regions  (should be run on a random catalogue)
    set_jackknife(dat, regmask, limits, noffset, njack, verbose=False) : Assign each object to jackknife regions (previously set up by solve_jackknife_nonsq())
    z_tozLG(dat) : transform redshifts from Heliocentric to LG frame
    ABSMAG(appmag,z,rest_GMR,kcorr_r,Qevol)  : Compute absolute magnitude taking into account k-correction and evolution parameterized by Qevol
    recompute_rest_col_mag(dat,regions, fsf, fresh=False, plot=True) : Assign restframe colours from g-r vs redshift lookup table and ABSMAG using k-correction polynomials
    compute_zmax_vmax(dat,regions)  :  Compute vmax and vmin using k-correction polynomials
    plot_kcorr(regions)  : plot k-correction polynomials
    plot_zmax_absmag(dat) : Plot how z_max depends on absolute magnitude and colour code by rest frame colour
    plot_zmin_absmag(dat) :  Plot how z_min depends on absolute magnitude and colour code by rest frame colour
    plot_zmax_z(dat)  :  Plot how z_max depends z colour code by absolute magnitude
    plot_zmin_z(dat)  :  Plot how z_min depends z colour code by absolute magnitude
    hist_nz(dat,ran,regions)  :  Plot the redshuft distribution dN/d
    plot_v_vmax(dat,regions)  :  plot the V/Vmax distribution for the selected sample
    plot_mag_z(dat,regions,contours=False)  :  magnitude-reshifts scatterplot
    plot_col_mag(dat,regions)  : colour magnitude scatter plot
    plot_col_mag_withvmax(dat,regions)  : colour magnitude scatter plot with objects weighted by vmax
    sky_plot(dat,regions) : All-sky scatter plot with galactic plane and ecliptic marked
    sky_plot_jack(dat)  : All-sky scatter plot with galactic plane and ecliptic marked with object jackknide regions in different colours
    cone_plot(dat,regions) :  This is hardwired to produce two particular cone plots but could be adapted
    lumfun_vmax(dat,regions, bandmag='ABSMAG_RP1', band='R', plot=True, saveplot=False, binwidth=0.25, ratio=False, Veff=False, Vollim=False) : 1/Vmax LF estimator
    lumfun_swml(dat,regions,log_phi_guess,magbins) : SWML LF estimator
    makefake(reg,nran)  : make a random catalogue.  **Needs to be checked**
    compute_veff(dat,regions) : compute Veff
    compute_veff_logspacing(dat,regions)  : Compute Veff using log-spaced distance bins (preferred)
    read_fsf(fpath) : Read John Moustakas' FastSpecFit catalogue
    remapfsfrmags(fsf)  : Force S catalogue colour distribution to agree with N
    load_full_catalogue_subsets(fullfile) : deprecated
    redhsift_slice(dat,zmin_sub,zmax_sub) : deprecated
    plot_sizes(dat,regions)               : deprecated
    plot_depths(dat,regions)              : deprecated
