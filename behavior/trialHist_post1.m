if ~isempty(B)
    sess_num = sess_num+1;
    dev_all(icount) = deviance;
    B_all(icount, :) = B;
    stats_all(icount) = stats;
    
    
    %% Do the conventional analysis to look at the effect of trial history for different ITIs.
    
    [fract_change_choosingSameChoice_aftS, fract_change_choosingSameChoice_aftF, ...
        fract_change_choosingHR_aftHR_vs_LR_S, fract_change_choosingLR_aftLR_vs_HR_S, ...
        fract_change_choosingHR_aftHR_vs_LR_F, fract_change_choosingLR_aftLR_vs_HR_F] = trialHist_0(...
        y, successPrevInput, failurePrevInput, itiPrecedInput, alldata, doiti, doplots, binningRates, vec_ratesdiff2, mouse);
    
    
    %%
    fract_change_choosingSameChoice_aftS_all =  [fract_change_choosingSameChoice_aftS_all; fract_change_choosingSameChoice_aftS];
    fract_change_choosingSameChoice_aftF_all = [fract_change_choosingSameChoice_aftF_all; fract_change_choosingSameChoice_aftF];
    fract_change_choosingHR_aftHR_vs_LR_S_all = [fract_change_choosingHR_aftHR_vs_LR_S_all; fract_change_choosingHR_aftHR_vs_LR_S];
    fract_change_choosingLR_aftLR_vs_HR_S_all = [fract_change_choosingLR_aftLR_vs_HR_S_all; fract_change_choosingLR_aftLR_vs_HR_S];
    fract_change_choosingHR_aftHR_vs_LR_F_all = [fract_change_choosingHR_aftHR_vs_LR_F_all; fract_change_choosingHR_aftHR_vs_LR_F];
    fract_change_choosingLR_aftLR_vs_HR_F_all = [fract_change_choosingLR_aftLR_vs_HR_F_all; fract_change_choosingLR_aftLR_vs_HR_F];
    
    
end
