% use this if you want to train the classifier lets say on correct trials
% and test it on incorrect trials. use numShuffs=1; Then run
% popClassifierSVM_traindata_set_plot.m
% with the following changes:


% you don't need : trsInChoiceVecUsedInIterS --> 
% make sure this is true though.

non_filtered = traces_al_sm(:, ~NsExcluded, :);
Y = allResp_HR_LR';

trs2use = (outcomes==0);


%%
% wNsHrLr_s(:, s);
% isequal(wNsHrLrAve_rep, wNsHrLr_s(:, s))

%
traces_bef_proj = non_filtered(:, :, trs2use);

frameTrProjOnBeta_rep = einsum(traces_bef_proj, wNsHrLrAve_rep, 2, 1);

frameTrProjOnBeta_hr_rep = frameTrProjOnBeta_rep(:, Y(trs2use)==1); % frames x trials % HR
frameTrProjOnBeta_lr_rep = frameTrProjOnBeta_rep(:, Y(trs2use)==0); % frames x trials % LR

frameTrProjOnBeta_hr_rep_all_alls_dataVSshuff_train{1}{1} = frameTrProjOnBeta_hr_rep;
frameTrProjOnBeta_lr_rep_all_alls_dataVSshuff_train{1}{1} = frameTrProjOnBeta_lr_rep;
%}

%%
