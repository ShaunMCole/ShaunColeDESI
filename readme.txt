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

Data files:
     ./data/fastspec-iron-main-bright.fits -- John Moustakas FastSpecFit catalogue
     ./data/colourLookupTable_N.fits       -- restframe colour look up table created from fastspec catalogue for North (Can be created by Augment_BGS_cat.ipynb)
     ./data/colourLookupTable_S.fits       -- restframe colour look up table created from fastspec catalogue for South (Can be created by Augment_BGS_cat.ipynb)