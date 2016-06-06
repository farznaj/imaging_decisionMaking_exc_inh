%% Compute threshold for separating HR and LR projections driven from the average decoder computed from all svm models.
% ie average of spikes in ep (trials x units) projected onto the decoder.
% (this decoder is computed by averaging the weights of decoders of all
% shuffles).

%% Get the average weights across all iters.

popClassifierSVM_traindata_setWeights


%% Compute threshold for separating HR from LR based on the projections of X (spike average during ep for actual trained data) onto the decoder (driven from all iters).

thresh = NaN(1, length(SVMModel_s_all));

% For each iteration get its x data, project it onto the averaged pooled
% weight of all iters. Then set the projections for hr vs lr, and compute
% the cross point of the 2 distributions assuming they are gaussian with
% the same sd.

% figure;
for s = 1:length(SVMModel_s_all)
    svm_model_now = SVMModel_s_all(s).shuff; % actual trained data
    
    x = svm_model_now.X;
    y = svm_model_now.Y;
    weights = w_norm_ave_train_dataVSshuff(:,1);
    
    weightedAveNs_allTrs = x * weights; % project on the average weights across all iters
    
    weightedAveNs_hr = weightedAveNs_allTrs(y==1);
    weightedAveNs_lr = weightedAveNs_allTrs(y==0);
    
    % Cross point of 2 gaussians when they have similar std:
    mu0 = mean(weightedAveNs_hr);
    mu1 = mean(weightedAveNs_lr);
    sd0 = std(weightedAveNs_hr);
    sd1 = std(weightedAveNs_lr);
    thresh(s) = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0);
   
%     nanmean(weightedAveNs_hr > thresh(s))
%     nanmean(weightedAveNs_lr < thresh(s))

%     subplot(2, 10-numShuffs/2, s), hold on
%     histogram(weightedAveNs_hr)
%     histogram(weightedAveNs_lr)
%     title(sprintf('%.2f', thresh(s)))

end

threshes_ave = nanmean(thresh);
cprintf('blue', '%.3f = Threshold for separating HR vs LR projections computed on training data for training window.\n', threshes_ave)
% figure; histogram(thresh)


%% Now compute classification accuracy for the temporal traces. 
% Get CV test trials (for each kfold and iter), and project their temporal
% traces onto weights (either weights of their own cv svm model, or pooled
% weights of all cv svm models). This (after pooling across kfolds and
% iters) gives traces_hr and traces_lr, (frames x trials). Then based on the threshes_ave
% decide what class each point belongs to. Then get the average across all
% trials to set class accuracy for all trials.

% usePooledWeights = 0 : project traces of test trials onto each cv models' own weights.
% usePooledWeights = 1 : project traces of test trials onto the normalized averaged pooled weight (of all kfolds and iters).

% id = 1;
% tracesHR = frameTrProjOnBeta_hr_rep_all_alls_dataVSshuff_cv{id};
% tracesLR = frameTrProjOnBeta_lr_rep_all_alls_dataVSshuff_cv{id};
traces_hr = cell2mat(tracesHR);
traces_lr = cell2mat(tracesLR);

sa = (traces_hr > threshes_ave);
sb = (traces_lr < threshes_ave);

corrClass_thresh = [sa, sb]; % frames x trials
av = nanmean(corrClass_thresh, 2);
sd = nanstd(corrClass_thresh, [], 2) / sqrt(size(corrClass_thresh,2));

% figure;
subplot(222)
boundedline(time_aligned, av, sd, 'c')


%{
% same as above
ab = [traces_hr, traces_lr];
p = ab > threshes_ave;
po = [ones(1, size(traces_hr,2)), zeros(1, size(traces_lr,2))];
m = bsxfun(@(x,y)(x==y), p, po);
figure; plot(time_aligned, nanmean(m, 2))
%}




%% Some checks

% figure; hold on
% histogram(traces_hr(:))
% histogram(traces_lr(:))
% 
% figure; hold on
% plot(time_aligned, nanmean(traces_hr,2))
% plot(time_aligned, nanmean(traces_lr,2))

mu0 = mean(traces_hr(:));
mu1 = mean(traces_lr(:));
sd0 = std(traces_hr(:));
sd1 = std(traces_lr(:));
th = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0);
fprintf('\t%.3f = threshold computed on HR vs LR temporal projections, all points \n', th)



% frame that classifier was trained on.
t = ep(ceil(length(ep)/2)); 
aa = cell2mat(cellfun(@(x)x(t,:), tracesHR, 'uniformoutput', 0));
bb = cell2mat(cellfun(@(x)x(t,:), tracesLR, 'uniformoutput', 0));
% figure; hold on
% histogram(aa(:))
% histogram(bb(:))

% Cross point of 2 gaussians when they have similar std:
mu0 = mean(aa(:));
mu1 = mean(bb(:));
sd0 = std(aa(:));
sd1 = std(bb(:));
th2 = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0);
fprintf('\t%.3f = threshold computed on HR vs LR temporal projections, training frame \n', th2)
%}


%% Compare average weights for svm models and cv svm models: they are quite close.
%{
figure; hold on
plot(w_norm_ave_train_dataVSshuff(:,1)) % normalized average of weights (from svm models) across all shuffles
plot(w_norm_ave_cv_dataVSshuff(:,1)) % normalized average of weights (from cv svm models) across all folds and shuffles

n = size(wNsHrLrAve_train_alls_dataVSshuff{1},1);
figure; hold on
boundedline(1:n, nanmean(wNsHrLrAve_train_alls_dataVSshuff{1}, 2), nanstd(wNsHrLrAve_train_alls_dataVSshuff{1}, [], 2))
boundedline(1:n, nanmean(wNsHrLrAve_cv_alls_dataVSshuff{1}, 2), nanstd(wNsHrLrAve_cv_alls_dataVSshuff{1}, [], 2), 'r')
%}

