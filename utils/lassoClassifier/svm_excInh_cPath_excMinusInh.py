# -*- coding: utf-8 -*-
"""
This script gets called in svm_excInh_cPath_plots.py

Created on Fri Jan  6 16:22:56 2017
@author: farznaj
"""


#%% For each day compute exc-inh for shuffle-averages. Also set %non0, weights, etc to nan when both exc and inh percent non-0 are 0 or 100.

perActiveEmI = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_non0 = np.full((perActiveAll_exc_aveS.shape), np.nan)
wEmI = np.full((perActiveAll_exc_aveS.shape), np.nan)
wEmI_non0 = np.full((perActiveAll_exc_aveS.shape), np.nan)

perActiveEmI_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wabsEmI_non0_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wEmI_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)
wEmI_non0_b0100nan = np.full((perActiveAll_exc_aveS.shape), np.nan)

for iday in days2ana: #range(len(days)):
    perActiveEmI[iday,:] = perActiveAll_exc_aveS[iday,:] - perActiveAll_inh_aveS[iday,:] # days x length(cvect_) # at each c value, divide %non-zero (averaged across shuffles) of exc and inh  
    wabsEmI[iday,:] = wall_abs_exc_aveNS[iday,:] - wall_abs_inh_aveNS[iday,:]
    wabsEmI_non0[iday,:] = wall_non0_abs_exc_aveNS[iday,:] - wall_non0_abs_inh_aveNS[iday,:]
    wEmI[iday,:] = wall_exc_aveNS[iday,:] - wall_inh_aveNS[iday,:]
    wEmI_non0[iday,:] = wall_non0_exc_aveNS[iday,:] - wall_non0_inh_aveNS[iday,:]
#    plt.plot(perActiveEmI[iday,:])

    # exclude c values for which both exc and inh %non-0 are 0 or 100.
    # you need to do +0, if not perActiveAll_exc_aveS will be also set to nan (same for w)
    a = perActiveAll_exc_aveS[iday,:]
    b = perActiveAll_inh_aveS[iday,:]    
    i0 = np.logical_and(a==0, b==0) # c values for which all weights of both exc and inh populations are 0.
    i100 = np.logical_and(a==100, b==100)  # c values for which all weights of both exc and inh populations are 100.
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    perActiveEmI_b0100nan[iday,:] = a - b
    
    a = wall_abs_exc_aveNS[iday,:]
    b = wall_abs_inh_aveNS[iday,:] 
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    wabsEmI_b0100nan[iday,:] = a - b
    
    a = wall_non0_abs_exc_aveNS[iday,:]
    b = wall_non0_abs_inh_aveNS[iday,:] 
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    wabsEmI_non0_b0100nan[iday,:] = a - b

    a = wall_exc_aveNS[iday,:]
    b = wall_inh_aveNS[iday,:] 
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    wEmI_b0100nan[iday,:] = a - b
    
    a = wall_non0_exc_aveNS[iday,:]
    b = wall_non0_inh_aveNS[iday,:] 
    a[i0] = np.nan
    b[i0] = np.nan
    a[i100] = np.nan
    b[i100] = np.nan
    wEmI_non0_b0100nan[iday,:] = a - b


    a = perClassErAll_aveS[iday,:] 
    a[i0] = np.nan
    a[i100] = np.nan
    
    
    a = perClassErTestAll_aveS[iday,:] 
    a[i0] = np.nan
    a[i100] = np.nan
    
    
#%% Set the null distribution for exc-inh comparison: For each day subtract random shuffles of exc (same for inh) from each other.

import numpy.random as rng
l = perActiveAll_exc.shape[1] # numSamples
randAll_perActiveAll_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # days x numSamples x length(cvect_) # exc - exc
randAll_perActiveAll_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_perActiveAll_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_wabs_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_wabs_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_wabs_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_wabs_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_wabs_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_wabs_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_w_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_w_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_w_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_w_exc_aveS = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_w_inh_aveS = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_w_ei_aveS = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh

randAll_perActiveAll_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # days x numSamples x length(cvect_) # exc - exc
randAll_perActiveAll_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_perActiveAll_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_wabs_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_wabs_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_wabs_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_wabs_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_wabs_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_wabs_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_w_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_w_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_w_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh
randAll_non0_w_exc_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # exc - exc
randAll_non0_w_inh_aveS_b0100nan = np.full(perActiveAll_exc.shape, np.nan) # inh - inh
randAll_non0_w_ei_aveS_b0100nan = np.full((perActiveAll_exc.shape[0],2*l,perActiveAll_exc.shape[2]), np.nan) # exc-exc and inh-inh

for iday in days2ana: #range(len(days)):
    
    ################## perc non-zero ######################
    # exc-exc
    inds = rng.permutation(l) # random indeces of shuffles
    inds1 = rng.permutation(l)
    a = perActiveAll_exc[iday,inds1,:] - perActiveAll_exc[iday,inds,:]
    randAll_perActiveAll_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = perActiveAll_exc[iday,inds1,:]# + 0 ... s
    bb = perActiveAll_exc[iday,inds,:]# + 0
    i0 = np.logical_and(aa==0, bb==0)
    i100 = np.logical_and(aa==100, bb==100)
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_perActiveAll_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    # inh-inh
    indsi = rng.permutation(l) # random indeces of shuffles
    inds1i = rng.permutation(l)
    b = perActiveAll_inh[iday,inds1i,:] - perActiveAll_inh[iday,indsi,:]
    randAll_perActiveAll_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = perActiveAll_inh[iday,inds1i,:]# + 0
    bb = perActiveAll_inh[iday,indsi,:]# + 0
    i0i = np.logical_and(aa==0, bb==0)
    i100i = np.logical_and(aa==100, bb==100)
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_perActiveAll_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    # pool exc - exc; inh - inh
    randAll_perActiveAll_ei_aveS[iday,:,:] = np.concatenate((a, b))  
#    plt.plot(np.mean(randAll_perActiveAll_exc_aveS[iday,:,:],axis=0))
    randAll_perActiveAll_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  
    
    
    
    ###################### w ######################
    # exc-exc
    # below uncommented, so %non-0 and ws are computed from the same set of shuffles.
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    a = wall_exc_aveN[iday,inds1,:] - wall_exc_aveN[iday,inds,:]
    randAll_w_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_exc_aveN[iday,inds1,:]
    bb = wall_exc_aveN[iday,inds,:]
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_w_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.

    
    # inh-inh
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    b = wall_inh_aveN[iday,inds1i,:] - wall_inh_aveN[iday,indsi,:]
    randAll_w_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_inh_aveN[iday,inds1i,:]
    bb = wall_inh_aveN[iday,indsi,:]
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_w_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    
    # pool exc - exc; inh - inh
    randAll_w_ei_aveS[iday,:,:] = np.concatenate((a, b))  
#    plt.plot(np.mean(randAll_perActiveAll_exc_aveS[iday,:,:],axis=0))
    randAll_w_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  

    
    ###################### non-0 w ######################
    # exc-exc
    # below uncommented, so %non-0 and ws are computed from the same set of shuffles.
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    a = wall_non0_exc_aveN[iday,inds1,:] - wall_non0_exc_aveN[iday,inds,:]
    randAll_non0_w_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_non0_exc_aveN[iday,inds1,:]
    bb = wall_non0_exc_aveN[iday,inds,:]
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_non0_w_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.

    
    # inh-inh
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    b = wall_non0_inh_aveN[iday,inds1i,:] - wall_non0_inh_aveN[iday,indsi,:]
    randAll_non0_w_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_non0_inh_aveN[iday,inds1i,:]
    bb = wall_non0_inh_aveN[iday,indsi,:]
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_non0_w_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    
    # pool exc - exc; inh - inh
    randAll_non0_w_ei_aveS[iday,:,:] = np.concatenate((a, b))  
#    plt.plot(np.mean(randAll_perActiveAll_exc_aveS[iday,:,:],axis=0))
    randAll_non0_w_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  


    
    ###################### abs w ######################
    # exc-exc
    # below uncommented, so %non-0 and ws are computed from the same set of shuffles.
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    a = wall_abs_exc_aveN[iday,inds1,:] - wall_abs_exc_aveN[iday,inds,:]
    randAll_wabs_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_abs_exc_aveN[iday,inds1,:]
    bb = wall_abs_exc_aveN[iday,inds,:]
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_wabs_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.

    
    # inh-inh
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    b = wall_abs_inh_aveN[iday,inds1i,:] - wall_abs_inh_aveN[iday,indsi,:]
    randAll_wabs_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_abs_inh_aveN[iday,inds1i,:]
    bb = wall_abs_inh_aveN[iday,indsi,:]
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_wabs_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    
    # pool exc - exc; inh - inh
    randAll_wabs_ei_aveS[iday,:,:] = np.concatenate((a, b))  
#    plt.plot(np.mean(randAll_perActiveAll_exc_aveS[iday,:,:],axis=0))
    randAll_wabs_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  
    
    
    
    ###################### abs non-0 w ######################
    # exc-exc
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    a = wall_non0_abs_exc_aveN[iday,inds1,:] - wall_non0_abs_exc_aveN[iday,inds,:]
    randAll_non0_wabs_exc_aveS[iday,:,:] = a # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    # you don't really need to do it for here, bc it is already non0 ws!
    aa = wall_non0_abs_exc_aveN[iday,inds1,:]
    bb = wall_non0_abs_exc_aveN[iday,inds,:]
    aa[i0] = np.nan
    bb[i0] = np.nan    
    aa[i100] = np.nan
    bb[i100] = np.nan    
    a2 = aa-bb
    randAll_non0_wabs_exc_aveS_b0100nan[iday,:,:] = a2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    

    # inh-inh
#    inds = rng.permutation(l) # random indeces of shuffles
#    inds1 = rng.permutation(l)
    b = wall_non0_abs_inh_aveN[iday,inds1i,:] - wall_non0_abs_inh_aveN[iday,indsi,:]
    randAll_non0_wabs_inh_aveS[iday,:,:] = b # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    # exclude c values for which both exc shuffle %non-0 are 0 or 100.
    aa = wall_non0_abs_inh_aveN[iday,inds1i,:]
    bb = wall_non0_abs_inh_aveN[iday,indsi,:]
    aa[i0i] = np.nan
    bb[i0i] = np.nan    
    aa[i100i] = np.nan
    bb[i100i] = np.nan    
    b2 = aa-bb
    randAll_non0_wabs_inh_aveS_b0100nan[iday,:,:] = b2 # numSamples x length(cvect_) ; subtract random shuff of exc from each other.
    
    
    # pool exc - exc; inh - inh
    randAll_non0_wabs_ei_aveS[iday,:,:] = np.concatenate((a, b))  
    randAll_non0_wabs_ei_aveS_b0100nan[iday,:,:] = np.concatenate((a2, b2))  
    
  