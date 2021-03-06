# -*- coding: utf-8 -*-
"""
Compare short vs long ITI SVM results after running each separately in mainSVM_notebook_plots.py
If looking at results without subselection of neurons, run svm_classAccur_noSubsel.py to get the vars.

Created on Sat Nov  5 14:36:16 2016
@author: farznaj
"""

#%% 

savefigs = 1


itiName = 'shortLong'
suffn = 'prev_%sITIs_%sN_' %(itiName, ntName)
suffnei = 'prev_%sITIs_%s_' %(itiName, 'excInh')
print '\n', suffn[:-1], ' - ', suffnei[:-1]


# part below commented... you used it for subsel (fni17)
"""    
#%% Set days that need be removed from short and long ITI data to make sure we are comparing the same days with each other
def shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, data, longiti, axn=1):
    in0not1 = fewRdays0[~np.in1d(fewRdays0, fewRdays1)]
    rmvfrom1 = np.array([daysOrig]).squeeze()[in0not1]
    
    in1not0 = fewRdays1[~np.in1d(fewRdays1, fewRdays0)]
    rmvfrom0 = np.array([daysOrig]).squeeze()[in1not0]
    
    if longiti==1:
        data = np.delete(data, np.argwhere(np.in1d(days1, rmvfrom1)), axis=axn)
    else:
        data = np.delete(data, np.argwhere(np.in1d(days0, rmvfrom0)), axis=axn)
        
    return data
    

#%% Set final days of analysis for both short and long ITI
fewRdays = np.union1d(fewRdays0, fewRdays1)

##%% Exclude days with all-0 weights (in all rounds) from analysis
'''
print 'Excluding %d days from analysis because all SVM weights of all rounds are 0' %(sum(all0d))
days = np.delete(days0, all0days)
'''
##%% Exclude days with only few (<=thR) rounds with non-0 weights from analysis
print 'Excluding %d days from analysis: they have <%d rounds with non-zero weights' %(len(fewRdays), thR)
days = np.delete(daysOrig, fewRdays)

numDays = len(days)
print 'Using', numDays, 'days for analysis:', days



#%%
'''
#####################################################################################################################################################   
############################ Classification accuracy at ep (testing data) ###################################################################################################     
#####################################################################################################################################################
'''

#%% Get common short-long-ITI days for all vars 
err_test_data_ave_allDays0 = shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, err_test_data_ave_allDays0, 0)
err_test_data_ave_allDays1 = shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, err_test_data_ave_allDays1, 1)


#%%
pall = np.full((len(days),2), np.nan)
for iday in range(len(days)):
    # subselect L1
#    _,p0 = stats.ttest_ind(err_test_data_ave_allDays0[:,iday], err_test_shfl_ave_allDays0[:,iday])
#    _,p1 = stats.ttest_ind(err_test_data_ave_allDays1[:,iday], err_test_shfl_ave_allDays1[:,iday])

    # L1 : get vars from svm_classAccur_l1_l2.py
    _,p0 = stats.ttest_ind(l1_err_test_data0[:,iday], l1_err_test_shfl0[:,iday], nan_policy='omit')
    _,p1 = stats.ttest_ind(l1_err_test_data1[:,iday], l1_err_test_shfl1[:,iday], nan_policy='omit')
    
    pall[iday,:] = [p0,p1]


numSigDays_short_longITI = np.sum(pall<=.05,axis=0) # number of sig days for short and long ITIs
print pall
print numSigDays_short_longITI 
""" 
 
#%% Average and std across rounds for subselect and across shuffles for L1
 # subselect L1
'''
ave_test_d0 = 100-np.nanmean(err_test_data_ave_allDays0, axis=0) # numDays
sd_test_d0 = np.nanstd(err_test_data_ave_allDays0, axis=0) 
ave_test_d1 = 100-np.nanmean(err_test_data_ave_allDays1, axis=0) # numDays
sd_test_d1 = np.nanstd(err_test_data_ave_allDays1, axis=0) 

ave_test_s0 = 100-np.nanmean(err_test_shfl_ave_allDays0, axis=0) # numDays
sd_test_s0 = np.nanstd(err_test_shfl_ave_allDays0, axis=0) 
ave_test_s1 = 100-np.nanmean(err_test_shfl_ave_allDays1, axis=0) # numDays
sd_test_s1 = np.nanstd(err_test_shfl_ave_allDays1, axis=0) 
'''

 # L1
ave_test_d0 = 100-np.nanmean(l1_err_test_data0, axis=0) # numDays
#sd_test_d0 = np.nanstd(l1_err_test_data0, axis=0) / np.sqrt(np.shape(l1_err_train_data0)[0])
sd_test_d0 = np.true_divide(np.nanstd(l1_err_test_data0, axis=0), np.sqrt(numNon0SampData0)) #/ np.sqrt(numSamp) 
ave_test_d1 = 100-np.nanmean(l1_err_test_data1, axis=0) # numDays
#sd_test_d1 = np.nanstd(l1_err_test_data1, axis=0) / np.sqrt(np.shape(l1_err_train_data0)[0]) 
sd_test_d1 = np.true_divide(np.nanstd(l1_err_test_data1, axis=0), np.sqrt(numNon0SampData1)) #/ np.sqrt(numSamp) 

ave_test_s0 = 100-np.nanmean(l1_err_test_shfl0, axis=0) # numDays
#sd_test_s0 = np.nanstd(l1_err_test_shfl0, axis=0) / np.sqrt(np.shape(l1_err_train_data0)[0]) 
sd_test_s0 = np.true_divide(np.nanstd(l1_err_test_shfl0, axis=0), np.sqrt(numNon0SampShfl0)) #/ np.sqrt(numSamp) 
ave_test_s1 = 100-np.nanmean(l1_err_test_shfl1, axis=0) # numDays
#sd_test_s1 = np.nanstd(l1_err_test_shfl1, axis=0) / np.sqrt(np.shape(l1_err_train_data0)[0]) 
sd_test_s1 = np.true_divide(np.nanstd(l1_err_test_shfl1, axis=0), np.sqrt(numNon0SampShfl1)) #/ np.sqrt(numSamp) 


_,p_shlo = stats.ttest_ind(ave_test_d0, ave_test_d1)
_,p_sh = stats.ttest_ind(ave_test_d0, ave_test_s0)
_,p_lo = stats.ttest_ind(ave_test_d1, ave_test_s1)
p_shlo, p_sh, p_lo


#%% you may want to do this ... excluding some days 
"""
# I think you better not do below bc when all weights are 0 it means no information was in the population... so shouldnt be removed
# exclude days to which <50 samples contributed (ie >=50 samples had all-0 weights)
ave_test_d0[numNon0SampData0<50] = np.nan
ave_test_d1[numNon0SampData1<50] = np.nan
ave_test_s0[numNon0SampData0<50] = np.nan
ave_test_s1[numNon0SampData1<50] = np.nan


# exclude non-sig days
'''
ave_test_d0[pall[:,0]>.05] = np.nan
ave_test_d1[pall[:,1]>.05] = np.nan
ave_test_s0[pall[:,0]>.05] = np.nan
ave_test_s1[pall[:,1]>.05] = np.nan
'''

# only take into account days that both short and longITI are sig:

a = np.sum(pall>.05,axis=1)!=0  # either IIT is insig (when comparing its data w shuffle)
ave_test_d0[a] = np.nan # either IIT is insig (when comparing its data w shuffle)
ave_test_d1[a] = np.nan
ave_test_s0[a] = np.nan
ave_test_s1[a] = np.nan

_,p_shlo = stats.ttest_ind(ave_test_d0, ave_test_d1, nan_policy='omit')
#_,p_sh = stats.ttest_ind(ave_test_d0, ave_test_s0, nan_policy='omit')
#_,p_lo = stats.ttest_ind(ave_test_d1, ave_test_s1, nan_policy='omit')
p_shlo#, p_sh, p_lo
"""

#%% Plot average across rounds for each day
plt.figure(figsize=(6,5))
gs = gridspec.GridSpec(2,5)#, width_ratios=[2, 1]) 

ax = plt.subplot(gs[0, 0:3])
plt.errorbar(range(numDays), ave_test_d0, yerr = sd_test_d0, color='r', label='Short ITI')
plt.errorbar(range(numDays), ave_test_d1, yerr = sd_test_d1, color='k', label='Long ITI')
#plt.errorbar(range(numDays), ave_test_s0, yerr = sd_test_s0, color='g', alpha=.35)
#plt.errorbar(range(numDays), ave_test_s1, yerr = sd_test_s1, color='k', alpha=.35)

plt.xlabel('Days')
plt.ylabel('Classification accuracy (%) - testing data')
plt.xlim([-1, len(days)])
lgd = plt.legend(loc='upper center', bbox_to_anchor=(.8,1.5), frameon=False)
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)


##%% Average across days : Data
x =[0,1]
labels = ['Short ITI', 'Long ITI']
ax = plt.subplot(gs[0,3:4])
plt.errorbar(x, [np.nanmean(ave_test_d0), np.nanmean(ave_test_d1)], yerr = [np.nanstd(ave_test_d0), np.nanstd(ave_test_d1)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
#plt.title('Data')
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical')    
ymin1, ymax = ax.get_ylim()
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
plt.title('Data\np= %.3f' %(p_shlo))

plt.subplots_adjust(wspace=1)
makeNicePlots(ax)



# Average across days : shuffle
x =[0,1]
labels = ['Short ITI', 'Long ITI']
ax = plt.subplot(gs[0,4:5])
plt.errorbar(x, [np.nanmean(ave_test_s0), np.nanmean(ave_test_s1)], yerr = [np.nanstd(ave_test_s0), np.nanstd(ave_test_s1)], marker='o', fmt=' ', color='k')
plt.xlim([x[0]-1, x[1]+1])
plt.title('Shuffle')
#plt.ylabel('Classification error (%) - testing data')
plt.xticks(x, labels, rotation='vertical')    
ymin2, _ = ax.get_ylim()
ymn = np.min([ymin1,ymin2])
plt.ylim([ymn, ymax])
#plt.tight_layout() #(pad=0.4, w_pad=0.5, h_pad=1.0)    
makeNicePlots(ax)

plt.subplot(gs[0,3:4])
plt.ylim([ymn, ymax])


# shuffle. all days
ax = plt.subplot(gs[1,0:3])
plt.errorbar(range(numDays), ave_test_s0, yerr = sd_test_s0, color='r', alpha=.35)
plt.errorbar(range(numDays), ave_test_s1, yerr = sd_test_s1, color='k', alpha=.35)

plt.xlabel('Days')
#plt.ylabel('Classification accuracy (%) - testing data')
plt.xlim([-1, len(days)])
#lgd = plt.legend(loc='upper center', bbox_to_anchor=(.8,1.3), frameon=False)
plt.title('Shuffle')
#leg.get_frame().set_linewidth(0.0)
makeNicePlots(ax)


plt.subplots_adjust(wspace=1, hspace=.7)


if savefigs:
#    dnow = '/shortLongAllITI_afterSFN/bestc_500Iters_non0decoder/setTo50ErrOfSampsWith0Weights' 
    fign = os.path.join(svmdir+dnow, suffn+'test_L1'+'.'+fmt[0])
    plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')


#%% Save the figure
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffn+'classError'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')
        
        
#%%   
'''
##################################################################################################################################################
########################### Projection Traces ####################################################################################################     
##################################################################################################################################################
'''

#%%
tr0_aligned0 = shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, tr0_aligned0, 0)
tr1_aligned0 = shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, tr1_aligned0, 0)
tr0_aligned1 = shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, tr0_aligned1, 1)
tr1_aligned1 = shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, tr1_aligned1, 1)

        
#%% Average across days

tr1_aligned0_ave = np.nanmean(tr1_aligned0, axis=1)
tr0_aligned0_ave = np.nanmean(tr0_aligned0, axis=1)
tr1_aligned0_std = np.nanstd(tr1_aligned0, axis=1)
tr0_aligned0_std = np.nanstd(tr0_aligned0, axis=1)

tr1_aligned1_ave = np.nanmean(tr1_aligned1, axis=1)
tr0_aligned1_ave = np.nanmean(tr0_aligned1, axis=1)
tr1_aligned1_std = np.nanstd(tr1_aligned1, axis=1)
tr0_aligned1_std = np.nanstd(tr0_aligned1, axis=1)

_,pproj0 = stats.ttest_ind(tr1_aligned0.transpose(), tr0_aligned0.transpose()) # p value of projections being different for hr vs lr at each time point
_,pproj1 = stats.ttest_ind(tr1_aligned1.transpose(), tr0_aligned1.transpose()) # p value of projections being different for hr vs lr at each time point


#%% Plot the average projections across all days
#ep_ms_allDays
plt.figure(figsize=(4.5,4))

ax = plt.subplot(211)
plt.fill_between(time_aligned, tr1_aligned0_ave - tr1_aligned0_std, tr1_aligned0_ave + tr1_aligned0_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_aligned0_ave, 'b', label = 'high rate')

plt.fill_between(time_aligned, tr0_aligned0_ave - tr0_aligned0_std, tr0_aligned0_ave + tr0_aligned0_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_aligned0_ave, 'r', label = 'low rate')

plt.xlabel('Time since stimulus onset')
plt.ylabel('SVM Projections')
plt.title('Short ITI')
plt.legend(loc='best', bbox_to_anchor=(1,1.2), frameon=False)

makeNicePlots(ax,0,1)

# Plot a dot for time points with significantly different hr and lr projections
ymin, ymax = ax.get_ylim()
pp = pproj0+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')


ax = plt.subplot(212)
plt.fill_between(time_aligned, tr1_aligned1_ave - tr1_aligned1_std, tr1_aligned1_ave + tr1_aligned1_std, alpha=0.5, edgecolor='b', facecolor='b')
plt.plot(time_aligned, tr1_aligned1_ave, 'b', label = 'High-rate choice')

plt.fill_between(time_aligned, tr0_aligned1_ave - tr0_aligned1_std, tr0_aligned1_ave + tr0_aligned1_std, alpha=0.5, edgecolor='r', facecolor='r')
plt.plot(time_aligned, tr0_aligned1_ave, 'r', label = 'Low-rate choice')

plt.xlabel('Time since stimulus onset')
plt.ylabel('Raw averages')
plt.title('Long ITI')

makeNicePlots(ax,0,1)
# Plot a dot for time points with significantly different hr and lr projections
ymin0, ymax0 = ax.get_ylim()
pp = pproj1+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')

plt.ylim([ymin, ymax])
plt.subplots_adjust(hspace=.8)

print 'ep_ms_allDays: \n', ep_ms_allDays



#%% Save the figure
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffn+'projTraces_svm_raw'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')


#%%
'''
########################################################################################################################################################       
#################### Classification accuracy at all times ##############################################################################################    
########################################################################################################################################################    
'''

#%%
corrClass_aligned0 = shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, corrClass_aligned0, 0)
corrClass_aligned1 = shortLongITISameDays(daysOrig, fewRdays0, fewRdays1, corrClass_aligned1, 1)


#%% Average across days

corrClass_aligned0_ave = np.mean(corrClass_aligned0, axis=1) * 100
corrClass_aligned0_std = np.std(corrClass_aligned0, axis=1) * 100 / float(np.sqrt(numDays))

corrClass_aligned1_ave = np.mean(corrClass_aligned1, axis=1) * 100
corrClass_aligned1_std = np.std(corrClass_aligned1, axis=1) * 100 / float(np.sqrt(numDays))

_,pcorrtrace01 = stats.ttest_ind(corrClass_aligned0.transpose(), corrClass_aligned1.transpose()) 

       
#%% Plot the average traces across all days
#ep_ms_allDays
plt.figure()

plt.fill_between(time_aligned, corrClass_aligned0_ave - corrClass_aligned0_std, corrClass_aligned0_ave + corrClass_aligned0_std, alpha=0.5, edgecolor='r', facecolor='r', label='Short ITI')
plt.plot(time_aligned, corrClass_aligned0_ave, 'r')

plt.xlabel('Time since stimulus onset (ms)')
plt.ylabel('Classification accuracy (%)')

ax = plt.gca()
makeNicePlots(ax)

# Plot a dot for significant time points
ymin, ymax = ax.get_ylim()
pp = pcorrtrace01+0; pp[pp>palpha] = np.nan; pp[pp<=palpha] = ymax
plt.plot(time_aligned, pp, color='k')

# Plot lines for the training epoch
win = np.mean(ep_ms_allDays, axis=0)
plt.plot([win[0], win[0]], [ymin, ymax], '-.', color=[.7, .7, .7])
plt.plot([win[1], win[1]], [ymin, ymax], '-.', color=[.7, .7, .7])


plt.fill_between(time_aligned, corrClass_aligned1_ave - corrClass_aligned1_std, corrClass_aligned1_ave + corrClass_aligned1_std, alpha=0.5, edgecolor='k', facecolor='k', label='Long ITI')
plt.plot(time_aligned, corrClass_aligned1_ave, 'k')


plt.legend(loc='best', frameon=False)


#%% Save the figure    
if savefigs:
    for i in range(np.shape(fmt)[0]): # save in all format of fmt         
        fign_ = suffn+'corrClassTrace'
        fign = os.path.join(svmdir, fign_+'.'+fmt[i])
        
        plt.savefig(fign, bbox_extra_artists=(lgd,), bbox_inches='tight')



        