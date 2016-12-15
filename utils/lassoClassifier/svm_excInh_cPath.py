# Main analysis to get c path plots for exc and inh neurons. To be run on the cluster.
# Once done, use svm_excInh_cPath_plots.py to plot figures.

# coding: utf-8


# NOTE: the following did not work on the cluster... I think you have to turn svm_excInh_setVars into a function to be able to use its vars in here. Alternatively just copy paste it here.
from svm_excInh_setVars import *
# Run mainSVM_notebook
execfile("svm_excInh_setVars.py")



#%%
##############################################################################################################################
# ## Excitatory and Inhibitory Neurons Relative Contribution to the decoder
# 
# We quantify the contribution of excitatory and inhibitory neurons to the encoding of the choice by measuring participation percentage, defined as the percentatge of a given population of neurons that has non-zero weights. We produce paraticipation curves, participation ratio at different values of svm regularizer (c), for each data
##############################################################################################################################


# In[292]:
'''
# This function finds the SVM decoder that predicts choices given responses in X by 
# using different values for c. At each value of c, it computes fraction of non-zero weights
# for exc and inh neurons, separately (perActive_inh, perActive_exc). Also it computes the 
# classification error (perClassEr) at each value of c. 
# Outputs: perActive_inh, perActive_exc, perClassEr, cvect_

def inh_exc_classContribution(X, Y, isinh): 
    import numpy as np
    from sklearn import svm
    
    def perClassError(Y, Yhat):
        import numpy as np
        perClassEr = sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
    
    Y = np.squeeze(Y); # class labels
    
    eps = 10**-10 # tiny number below which weight is considered 0
    isinh = isinh>0;  # vector of size number of neurons (1: neuron is inhibitory, 0: neuron is excitatoey); here I am making sure to convert it to logical
    n_inh = sum(isinh);
    n_exc = sum(~ isinh);
    cvect_ = 10**(np.arange(-4, 6,0.1))/len(Y);
    perClassEr = [];
    perActive_inh = [];
    perActive_exc = [];
    w_allc = []
    for i in range(len(cvect_)): # At each value of cvect we compute the fraction of non-zero weights for excit and inhibit neurons.
        linear_svm = [];
        linear_svm = svm.LinearSVC(C = cvect_[i], loss='squared_hinge', penalty='l1', dual=False);
        linear_svm.fit(X, Y);
        w = np.squeeze(linear_svm.coef_);
        
        perActive_inh.append(sum(abs(w[isinh])>eps)/ (n_inh + 0.) * 100.)
        perActive_exc.append(sum(abs(w[~isinh])>eps)/ (n_exc + 0.) * 100.)
        w_allc.append(w) # includes weights of all neurons for each value of c
        perClassEr.append(perClassError(Y, linear_svm.predict(X)));
    
    return perActive_inh, perActive_exc, perClassEr, cvect_, w_allc
'''

#%%
# This function finds the SVM decoder that predicts choices given responses in X by 
# using different values for c. At each value of c, it computes fraction of non-zero weights
# for exc and inh neurons, separately (perActive_inh, perActive_exc). Also it computes the 
# classification error (perClassEr) at each value of c. 
# Outputs: perActive_inh, perActive_exc, perClassEr, cvect_

def inh_exc_classContribution(X, Y, isinh): 
    import numpy as np
    from sklearn import svm
    from crossValidateModel import crossValidateModel
    from linearSVM import linearSVM
        
    def perClassError(Y, Yhat):
        import numpy as np
        perClassEr = sum(abs(np.squeeze(Yhat).astype(float)-np.squeeze(Y).astype(float)))/len(Y)*100
        return perClassEr
    
    Y = np.squeeze(Y); # class labels    
    eps = 10**-10 # tiny number below which weight is considered 0
    isinh = isinh>0;  # vector of size number of neurons (1: neuron is inhibitory, 0: neuron is excitatoey); here I am making sure to convert it to logical
    n_inh = sum(isinh);
    n_exc = sum(~ isinh);
#     cvect_ = 10**(np.arange(-4, 6,0.1))/len(Y);
    cvect_ = 10**(np.arange(-6.5, 3.5, 0.1)) # FN: use this if you want the same cvect for all days

    numShuffles_ei = 100 # 100 times we subselect trials as training and testing datasetes.    
    regType = 'l1'
    kfold = 10;
    perClassErrorTrain_data_ei = np.full((numShuffles_ei, len(cvect_)), np.nan)
    perClassErrorTest_data_ei = np.full((numShuffles_ei, len(cvect_)), np.nan)
    w_data_ei = np.full((numShuffles_ei, len(cvect_), n_inh+n_exc), np.nan)
    b_data_ei = np.full((numShuffles_ei, len(cvect_)), np.nan)
    perActive_exc_data_ei = np.full((numShuffles_ei, len(cvect_)), np.nan)
    perActive_inh_data_ei = np.full((numShuffles_ei, len(cvect_)), np.nan)
    
    for i in range(len(cvect_)): # At each value of cvect we compute the fraction of non-zero weights for excit and inhibit neurons.
        summary_data_ei = [];     
        for ii in range(numShuffles_ei): # generate random training and testing datasets
            if regType == 'l1':
                summary_data_ei.append(crossValidateModel(X, Y, linearSVM, kfold = kfold, l1 = cvect_[i]))
            elif regType == 'l2':
                summary_data_ei.append(crossValidateModel(X, Y, linearSVM, kfold = kfold, l2 = cvect_[i]))

            w = np.squeeze(summary_data_ei[ii].model.coef_);
            perClassErrorTrain_data_ei[ii,i] = summary_data_ei[ii].perClassErrorTrain
            perClassErrorTest_data_ei[ii,i] = summary_data_ei[ii].perClassErrorTest
            w_data_ei[ii,i,:] = w;
            b_data_ei[ii,i] = summary_data_ei[ii].model.intercept_;
            perActive_inh_data_ei[ii,i] = sum(abs(w[isinh])>eps)/ (n_inh + 0.) * 100.
            perActive_exc_data_ei[ii,i] = sum(abs(w[~isinh])>eps)/ (n_exc + 0.) * 100.
    
    # Do this: here average values across trialShuffles to avoid very large vars
    '''
    perClassErrorTrain_data_ei = np.mean(perClassErrorTrain_data_ei, axis=0)
    perClassErrorTest_data_ei = np.mean(perClassErrorTest_data_ei, axis=0)
    w_data_ei = np.mean(w_data_ei, axis=0)
    b_data_ei = np.mean(b_data_ei, axis=0)
    perActive_inh_data_ei = np.mean(perActive_inh_data_ei, axis=0)
    perActive_exc_data_ei = np.mean(perActive_exc_data_ei, axis=0)
    '''
    return perActive_inh_data_ei, perActive_exc_data_ei, perClassErrorTrain_data_ei, perClassErrorTest_data_ei, cvect_, w_data_ei, b_data_ei 


# In[293]:

if neuronType==2:
#    perActive_inh_allExc, perActive_exc_allExc, perClassEr_allExc, cvect_, wei_all_allExc = inh_exc_classContribution(X[:, ~np.isnan(inhRois)], Y, inhRois[~np.isnan(inhRois)])
    perActive_inh_allExc, perActive_exc_allExc, perClassEr_allExc, perClassErTest_allExc, cvect_, wei_all_allExc, bei_all_allExc = inh_exc_classContribution(X[:, ~np.isnan(inhRois)], Y, inhRois[~np.isnan(inhRois)])

# In[308]:

# Plot average of all weights, average of non-zero weights, and percentage of non-zero weights for each value of c
# Training the classifier using all exc and inh neurons at different values of c.

if doPlots and neuronType==2:    
    wei_all_allExc = np.array(wei_all_allExc)
    # plot ave across neurons for each value of c
    inhRois_allExc = inhRois[~np.isnan(inhRois)]
    
    
    ########
    # average and std of weights across neurons
    plt.figure(figsize=(4,3))
    plt.errorbar(cvect_, np.mean(wei_all_allExc[:,inhRois_allExc==0],axis=1), np.std(wei_all_allExc[:,inhRois_allExc==0],axis=1), color='b', label='excit')
    plt.errorbar(cvect_, np.mean(wei_all_allExc[:,inhRois_allExc==1],axis=1), np.std(wei_all_allExc[:,inhRois_allExc==1],axis=1), color='r', label='inhibit')
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
#     plt.ylim([-10,110])
    plt.legend(loc='center left', bbox_to_anchor=(1, .7))
    plt.ylabel('Average of weights')
    
    
    ########
    # Average and std of non-zero weights across neurons
    wei_all_0inds = np.array([x==0 for x in wei_all_allExc]) # inds of zero weights
    wei_all_non0 = wei_all_allExc+0
    wei_all_non0[wei_all_0inds] = np.nan # set 0 weights to nan
    
    plt.figure(figsize=(4,3))
    plt.errorbar(cvect_, np.nanmean(wei_all_non0[:,inhRois_allExc==0],axis=1), np.nanstd(wei_all_non0[:,inhRois_allExc==0],axis=1), color='b', label='excit')
    plt.errorbar(cvect_, np.nanmean(wei_all_non0[:,inhRois_allExc==1],axis=1), np.nanstd(wei_all_non0[:,inhRois_allExc==1],axis=1), color='r', label='inhibit')
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
#     plt.ylim([-10,110])
    plt.legend(loc='center left', bbox_to_anchor=(1, .7))
    plt.ylabel('Average of non-zero weights')
    
    
    ########
    # Percentage of non-zero weights
    wei_all_0inds = np.array([x==0 for x in wei_all_allExc]) # inds of zero weights
#     percNonZero_e = np.mean(wei_all_0inds[:,inhRois_ei==0]==0, axis=1) # fraction of nonzero weights per round and per c
#     percNonZero_i = np.mean(wei_all_0inds[:,inhRois_ei==1]==0, axis=1)
        
    plt.figure(figsize=(4,3))
    plt.plot(cvect_, perClassEr_allExc, 'k-', label = '% classification error')
    plt.plot(cvect_, 100*np.nanmean(wei_all_0inds[:,inhRois_allExc==0]==0,axis=1), color='b', label='excit')
    plt.plot(cvect_, 100*np.nanmean(wei_all_0inds[:,inhRois_allExc==1]==0,axis=1), color='r', label='inhibit')    
#     plt.errorbar(cvect_, np.nanmean(wei_all_0inds[:,inhRois_allExc==0]==0,axis=1), np.nanstd(wei_all_0inds[:,inhRois_allExc==0],axis=1), color='b', label='excit')
#     plt.errorbar(cvect_, np.nanmean(wei_all_0inds[:,inhRois_allExc==1]==0,axis=1), np.nanstd(wei_all_0inds[:,inhRois_allExc==1],axis=1), color='r', label='inhibit')
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
#     plt.ylim([-10,110])
    plt.legend(loc='center left', bbox_to_anchor=(1, .7))
    plt.ylabel('% non-zero weights')
    
    '''
#     if doPlots and neuronType==2:    
    plt.figure(figsize=(4,3))
#     plt.subplot(221)
    plt.plot(cvect_, perActive_exc_allExc, 'b-', label = 'excit % non-zero w')
    plt.plot(cvect_, perActive_inh_allExc, 'r-', label = 'inhibit % non-zero w')
    plt.plot(cvect_, perClassEr_allExc, 'k-', label = '% classification error')
    plt.xscale('log')
    plt.xlabel('c (inverse of regularization parameter)')
    plt.ylim([-10,110])
    plt.legend(loc='center left', bbox_to_anchor=(1, .7))
    '''



#%%
######################################################################################################################################################
######################################################################################################################################################
#  ### Another version of the analysis: equal number of exc and inh neurons
#  We control for the different numbers of excitatory and inhibitory neurons by subsampling n excitatory neurons, where n is equal to the number of inhibitory neurons. More specifically, instead of sending the entire X to the function inh_exc_classContribution, we use a subset of X that includes equal number of exc and inh neurons (Exc neurons are randomly selected).
######################################################################################################################################################
######################################################################################################################################################

# In[315]:

# Use equal number of exc and inh neurons
if (neuronType==2 and not 'w' in locals()) or (neuronType==2 and 'w' in locals() and np.sum(w)!=0):
    X_ = X[:, ~np.isnan(inhRois)];
    inhRois_ = inhRois[~np.isnan(inhRois)].astype('int32')
    ix_i = np.argwhere(inhRois_).squeeze()
    ix_e = np.argwhere(inhRois_-1).squeeze()
    n = len(ix_i);
    Xei = np.zeros((len(Y), 2*n));
    inhRois_ei = np.zeros((2*n));
    
    perActive_inh = [];
    perActive_exc = [];
    perClassEr = [];
    wei_all = []
    
    bei_all = []
    perClassErTest = []
    
    for i in range(numSamples): # shuffle neurons
        msk = rng.permutation(ix_e)[0:n];
        Xei[:, 0:n] = X_[:, msk]; # first n columns are X of random excit neurons.
        inhRois_ei[0:n] = 0;

        Xei[:, n:2*n] = X_[:, ix_i]; # second n icolumns are X of inhibit neurons.
        inhRois_ei[n:2*n] = 1;
        
        # below we fit svm onto Xei, which for all shuffles (numSamples) has the same set of inh neurons but different sets of exc neurons
#         ai, ae, ce, cvect_, wei = inh_exc_classContribution(Xei, Y, inhRois_ei); # ai is of length cvect defined in inh_exc_classContribution
        ai, ae, ce, ces, cvect_, wei, bei = inh_exc_classContribution(Xei, Y, inhRois_ei)

        perActive_inh.append(ai); # numSamples x numTrialShuff x length(cvect_)
        perActive_exc.append(ae); # numSamples x numTrialShuff x length(cvect_)
        wei_all.append(wei) # numSamples x numTrialShuff x length(cvect_) x numNeurons(inh+exc equal numbers)        
        perClassEr.append(ce); # numSamples x numTrialShuff x length(cvect_)      # numTrialShuff is number of times you shuffled trials to do cross validation. (it is variable numShuffles_ei in function inh_exc_classContribution)
        
        bei_all.append(bei)  # numSamples x numTrialShuff x length(cvect_) 
        perClassErTest.append(ces); perClassErTest
            


# In[285]:

if (neuronType==2 and not 'w' in locals()) or (neuronType==2 and 'w' in locals() and np.sum(w)!=0):
    
    # p value of comparing exc and inh non-zero weights pooled across values of c :
    aa = np.array(perActive_exc).flatten()
#     aa = aa[np.logical_and(aa>0 , aa<100)]
    # np.shape(aa)

    bb = np.array(perActive_inh).flatten()
#     bb = bb[np.logical_and(bb>0 , bb<100)]
    # np.shape(bb)

    h, p_two = stats.ttest_ind(aa, bb)
    p_tl = ttest2(aa, bb, tail='left')
    p_tr = ttest2(aa, bb, tail='right')
    print '\np value (pooled for all values of c):\nexc ~= inh : %.2f\nexc < inh : %.2f\nexc > inh : %.2f' %(p_two, p_tl, p_tr)


    # two-tailed p-value
    h, p_two = stats.ttest_ind(np.array(perActive_exc), np.array(perActive_inh))
    # left-tailed p-value : excit < inhibit
    p_tl = ttest2(np.array(perActive_exc), np.array(perActive_inh), tail='left')
    # right-tailed p-value : excit > inhibit
    p_tr = ttest2(np.array(perActive_exc), np.array(perActive_inh), tail='right')
    
    
    
    # Plot the c path :
    if doPlots:
        plt.figure(figsize=(4,3)) 
        plt.errorbar(cvect_, np.mean(np.array(perActive_exc), axis=0), yerr=2*np.std(np.array(perActive_exc), axis=0), color = 'b', label = 'excit % non-zero weights')
        plt.errorbar(cvect_, np.mean(np.array(perActive_inh), axis=0), yerr=2*np.std(np.array(perActive_inh), axis=0), color = 'r', label = 'inhibit % non-zero weights')
        plt.xscale('log')
        plt.ylim([-10,110])
        # plt.ylabel('% non-zero weights')
        # plt.legend(loc='center left', bbox_to_anchor=(1, .7))
        plt.xlim([cvect_[0], cvect_[-1]])

        # plt.plot(cvect_, p_two, label = 'excit ~= inhibit')
        # plt.plot(cvect_, p_tl, label = 'excit < inhibit')
        # plt.plot(cvect_, p_tr, label = 'inhibit < excit')


        # plt.figure()
        plt.errorbar(cvect_ ,np.mean(np.array(perClassEr), axis=0), yerr=2*np.std(np.array(perClassEr), axis=0), color = 'k', label = 'classification error') # range(len(perClassEr[0]))
        plt.ylim([-10,110])
        plt.xscale('log')
        plt.xlabel('c (inverse of regularization parameter)')
        # plt.ylabel('classification error')
        plt.xlim([cvect_[0], cvect_[-1]])
        plt.legend(loc='center left', bbox_to_anchor=(1, .7))
        # plt.tight_layout()


        plt.figure(figsize=(4,3))
#        plt.subplot(212)
        # plt.plot(cvect_, p_two<.05, label = 'excit ~= inhibit')
        # plt.plot(cvect_, p_tl<.05, label = 'excit < inhibit')
        # plt.plot(cvect_, p_tr<.05, label = 'inhibit < excit')
        plt.plot(cvect_, p_two, label = 'excit ~= inhibit')
        plt.plot(cvect_, p_tl, label = 'excit < inhibit')
        plt.plot(cvect_, p_tr, label = 'inhibit < excit')
        plt.xscale('log')
        plt.ylim([-.1, 1.1])
        # plt.legend(loc='center left', bbox_to_anchor=(1, .7))
        plt.ylabel('P value')
        plt.xlim([cvect_[0], cvect_[-1]])
        plt.xlabel('c (inverse of regularization parameter)')

        plt.figure(figsize=(4,3))
#        plt.subplot(211)
        plt.plot(cvect_, p_two<.05, label = 'excit ~= inhibit')
        plt.plot(cvect_, p_tl<.05, label = 'excit < inhibit')
        plt.plot(cvect_, p_tr<.05, label = 'excit > inhibit')
        plt.xscale('log')
        plt.ylim([-.1, 1.1])
        plt.xlim([cvect_[0], cvect_[-1]])
        plt.ylabel('P value < .05')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7))


# In[319]:

# Plot average of all weights, average of non-zero weights, and percentage of non-zero weights for each value of c
# Training the classifier using all exc and inh neurons at different values of c.

if doPlots and (neuronType==2 and not 'w' in locals()) or (neuronType==2 and 'w' in locals() and np.sum(w)!=0):
    
    xaxisErr = 0; # if 1 x axis will be training error, otherwise it will be c.
    
    wei_all = np.array(wei_all)
    
    ########
    # average of weights    

    # Average weights of all neurons for each value of c. Then plot average and std across rounds for each value of c.

    # exc
    wave = np.mean(wei_all[:,:,inhRois_ei==0], axis=2) # average of all neural weights per c value and per round
    ave = np.mean(wave, axis=0) # average of weights across rounds
    sde = np.std(wave, axis=0) # std of weights across rounds

    # inh
    wavi = np.mean(wei_all[:,:,inhRois_ei==1], axis=2) # average of all neural weights per c value and per round
    avi = np.mean(wavi, axis=0) # average of weights across rounds
    sdi = np.std(wavi, axis=0) # std of weights across rounds


    aveEr = np.mean(np.array(perClassEr), axis=0)
    if xaxisErr:
        x = aveEr
    else:
        x = cvect_
        
    plt.figure(figsize=(4,3))
    plt.errorbar(x, ave, sde, color = 'b', label = 'excit')
    plt.errorbar(x, avi, sdi, color = 'r', label = 'inhibit')
    
    if xaxisErr:
        plt.xlabel('Training error %')
    else:
        plt.xscale('log')
        plt.xlim([x[0], x[-1]])
        plt.xlabel('c (inverse of regularization parameter)')

    plt.legend(loc=0)
    plt.ylabel('Average of weights')




    ########
    # Average non-zero weights
    wei_all_0inds = np.array([x==0 for x in wei_all]) # inds of zero weights
    wei_all_non0 = wei_all+0
    wei_all_non0[wei_all_0inds] = np.nan # set 0 weights to nan
    # wei_all_non0.shape


    # Average non-zero weights of all neurons for each value of c. Then plot average and std across rounds for each value of c.

    # exc
    wave = np.nanmean(wei_all_non0[:,:,inhRois_ei==0], axis=2) # average of all neural weights per c value and per round
    ave = np.nanmean(wave, axis=0) # average of weights across rounds
    sde = np.nanstd(wave, axis=0) # std of weights across rounds

    # inh
    wavi = np.nanmean(wei_all_non0[:,:,inhRois_ei==1], axis=2) # average of all neural weights per c value and per round
    avi = np.nanmean(wavi, axis=0) # average of weights across rounds
    sdi = np.nanstd(wavi, axis=0) # std of weights across rounds


    if xaxisErr:
        x = aveEr
    else:
        x = cvect_

    plt.figure(figsize=(4,3))
    plt.errorbar(x, ave, sde, color = 'b', label = 'excit')
    plt.errorbar(x, avi, sdi, color = 'r', label = 'inhibit')
    
    if xaxisErr:
        plt.xlabel('Training error %')
    else:
        plt.xscale('log')
        plt.xlim([x[0], x[-1]])
        plt.xlabel('c (inverse of regularization parameter)')

    plt.legend(loc=0)
    plt.ylabel('Average of non-zero weights')




    ########
    # Percentage of non-zero weights

    wei_all_0inds = np.array([x==0 for x in wei_all]) # inds of zero weights
    percNonZero_e = 100*np.mean(wei_all_0inds[:,:,inhRois_ei==0]==0, axis=2) # fraction of nonzero weights per round and per c
    percNonZero_i = 100*np.mean(wei_all_0inds[:,:,inhRois_ei==1]==0, axis=2)

    # Average of percentage of non-zero weights for each value of c across rounds.

    # exc
    ave = np.nanmean(percNonZero_e, axis=0) # average of weights across rounds
    sde = np.nanstd(percNonZero_e, axis=0) # std of weights across rounds

    # inh
    avi = np.nanmean(percNonZero_i, axis=0) # average of weights across rounds
    sdi = np.nanstd(percNonZero_i, axis=0) # std of weights across rounds


    if xaxisErr:
        x = aveEr
    else:
        x = cvect_

    plt.figure(figsize=(4,3))
    plt.errorbar(x ,np.mean(np.array(perClassEr), axis=0), np.std(np.array(perClassEr), axis=0), color = 'k', label = 'classification error') # range(len(perClassEr[0]))
    plt.errorbar(x, ave, sde, color = 'b', label = 'excit')
    plt.errorbar(x, avi, sdi, color = 'r', label = 'inhibit')
    
    if xaxisErr:
        plt.xlabel('Training error %')
    else:
        plt.xscale('log')
        plt.xlim([x[0], x[-1]])
        plt.xlabel('c (inverse of regularization parameter)')

    plt.xlabel('Training error %')
    plt.legend(loc='upper left', bbox_to_anchor=(1, .7))
    plt.ylabel('% non-zero weights')


# ## Save results as .mat files in a folder named svm

# In[44]:

if trialHistAnalysis:
#     ep_ms = np.round((ep-eventI)*frameLength)
    th_stim_dur = []
    svmn = 'excInhC_svmPrevChoice_%sN_%sITIs_ep%d-%dms_%s_' %(ntName, itiName, ep_ms[0], ep_ms[-1], nowStr)
else:
    svmn = 'excInhC_svmCurrChoice_%sN_ep%d-%dms_%s_' %(ntName, ep_ms[0], ep_ms[-1], nowStr)   
print '\n', svmn[:-1]

if saveResults:
    print 'Saving .mat file'
    d = os.path.join(os.path.dirname(pnevFileName), 'svm')
    if not os.path.exists(d):
        print 'creating svm folder'
        os.makedirs(d)

    svmName = os.path.join(d, svmn+os.path.basename(pnevFileName))
    print(svmName)
    # scio.savemat(svmName, {'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'meanX':meanX, 'stdX':stdX, 'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'th_stim_dur':th_stim_dur})
    if neuronType==2:
#         scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'NsRand':NsRand, 'meanX':meanX, 'stdX':stdX, 'w':w, 'b':b, 'cbest':cbest, 'corrClass':corrClass, 'perClassErrorTrain_data':perClassErrorTrain_data, 'perClassErrorTrain_shfl':perClassErrorTrain_shfl, 'perClassErrorTest_data':perClassErrorTest_data, 'perClassErrorTest_shfl':perClassErrorTest_shfl, 'perClassErrorTest':perClassErrorTest, 'perClassErrorTrain':perClassErrorTrain, 'cvect':cvect, 'perActive_inh':perActive_inh, 'perActive_exc':perActive_exc, 'perClassEr':perClassEr, 'cvect_':cvect_, 'trainE':trainE, 'train_err_exc0':train_err_exc0, 'train_err_inh0':train_err_inh0, 'corrClass_exc0':corrClass_exc0, 'corrClass_inh0':corrClass_inh0, 'train_err_allExc0':train_err_allExc0, 'corrClass_allExc0':corrClass_allExc0})
        scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 
                               'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 
                               'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 
                               'NsRand':NsRand, 'meanX':meanX, 'stdX':stdX, 
                               'perActive_inh':perActive_inh, 'perActive_exc':perActive_exc, 
                               'perClassEr':perClassEr, 'wei_all':wei_all, 
                               'perActive_inh_allExc':perActive_inh_allExc, 'perActive_exc_allExc':perActive_exc_allExc, 
                               'perClassEr_allExc':perClassEr_allExc, 'wei_all_allExc':wei_all_allExc,
                               'bei_all_allExc':bei_all_allExc, 'perClassErTest_allExc':perClassErTest_allExc,
                               'bei_all':bei_all, 'perClassErTest':perClassErTest,
                               'cvect_':cvect_}) 
    
#    else:
#        scio.savemat(svmName, {'thAct':thAct, 'thTrsWithSpike':thTrsWithSpike, 'ep_ms':ep_ms, 'th_stim_dur':th_stim_dur, 'numSamples':numSamples, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'NsRand':NsRand, 'meanX':meanX, 'stdX':stdX, 'w':w, 'b':b, 'cbest':cbest, 'corrClass':corrClass, 'perClassErrorTrain_data':perClassErrorTrain_data, 'perClassErrorTrain_shfl':perClassErrorTrain_shfl, 'perClassErrorTest_data':perClassErrorTest_data, 'perClassErrorTest_shfl':perClassErrorTest_shfl, 'perClassErrorTest':perClassErrorTest, 'perClassErrorTrain':perClassErrorTrain, 'cvect':cvect})

    # save normalized traces as well                       
    # scio.savemat(svmName, {w':w, 'b':b, 'cbest':cbest, 'corrClass':corrClass, 'trsExcluded':trsExcluded, 'NsExcluded':NsExcluded, 'meanX':meanX, 'stdX':stdX, 'X':X, 'Y':Y, 'Xt':Xt, 'Xtg':Xtg, 'Xtc':Xtc, 'Xtr':Xtr, 'Xtp':Xtp})
    # 'linear_svm':linear_svm, 

    # append : doesn't quite work
    # if os.path.isfile(svmName): 
    #     with open(svmName,'ab') as f:
    #         sci.io.savemat(f, {'perClassErrorTrain_data':perClassErrorTrain_data, 'perClassErrorTrain_shfl':perClassErrorTrain_shfl, 'perClassErrorTest_data':perClassErrorTest_data, 'perClassErrorTest_shfl':perClassErrorTest_shfl}) # append
    # else:
else:
    print 'Not saving .mat file'
    
    
# If you want to save the linear_svm objects as mat file, you need to take care of 
# None values and set to them []:

# import inspect
# inspect.getmembers(summary_shfl_exc[0])

# for i in range(np.shape(summary_shfl_exc)[0]):
#     summary_shfl_exc[i].model.random_state = []
#     summary_shfl_exc[i].model.class_weight = []
# [summary_shfl_exc[i].model.random_state for i in range(np.shape(summary_shfl_exc)[0])]
# [summary_shfl_exc[i].model.class_weight for i in range(np.shape(summary_shfl_exc)[0])]

# scio.savemat(svmName, {'summary_shfl_exc':summary_shfl_exc})

# Data = scio.loadmat(svmName, variable_names=['summary_shfl_exc'] ,squeeze_me=True,struct_as_record=False)
# summary_shfl_exc = Data.pop('summary_shfl_exc')
# summary_shfl_exc[0].model.intercept_
    


# ## Move the autosaved .html file to a folder named "figs"
#     Notebook html file will be autosaved in the notebook directory if jupyter_notebook_config.py exists in ~/.jupyter and includes the function script_post_save. Below we move the html to a directory named figs inside the root directory which contains moreFile, etc.

# In[45]:

# make sure autosave is done so you move the most recent html file to figs directory.
if 'ipykernel' in sys.modules and saveHTML:
    get_ipython().magic(u'autosave 1')
    # import time    
    # time.sleep(2) 
    
    d = os.path.join(os.path.dirname(pnevFileName),'figs')
    if not os.path.exists(d):
        print 'creating figs folder'
        os.makedirs(d)

    htmlName = os.path.join(d, svmn[:-1]+'.html')
    print htmlName
    import shutil
    shutil.move(os.path.join(os.getcwd(), 'mainSVM_notebook.html'), htmlName)

    # go back to default autosave interval
    get_ipython().magic(u'autosave 120')

