% make some plots related to the decoder

%% Simple averaging of neural responses for HR and LR trials. 
% decide you training window based on this plot!

% traces = traces_al_sm(:, ~NsExcluded, :); % equal number of trs for both conds
% av1 = nanmean(nanmean(traces(:,:, choiceVec==1), 3), 2);
% av2 = nanmean(nanmean(traces(:,:, choiceVec==0), 3), 2);

traces = traces_al_sm(:, ~NsExcluded, :); % analyze all trials
av1 = nanmean(nanmean(traces(:,:, choiceVec0==1), 3), 2);
av2 = nanmean(nanmean(traces(:,:, choiceVec0==0), 3), 2);
% average for HR vs LR stimulus.
% av1 = nanmean(nanmean(traces(:,:, stimrate > cb), 3), 2);
% av2 = nanmean(nanmean(traces(:,:, stimrate < cb), 3), 2);

pb = round((- eventI_stimOn)*frameLength);
pe = round((size(traces,1) - eventI_stimOn)*frameLength);
st = round((epStart - eventI_stimOn)*frameLength);
en = round((epEnd - eventI_stimOn)*frameLength);
mn = min([av1;av2]);
mx = max([av1;av2]);
figure; 
subplot(223), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot(time_aligned_stimOn, av1), plot(time_aligned_stimOn, av2)
xlabel('Time since stim onset (ms)')
ylabel('Average neural responses')
xlim([pb pe])


%% Weighted average of neurons for each trial, using svm weights trained on a particular epoch (ep).

% see how well that particular epoch can decode other epochs.

% include all trs (not just the random equal trs) and use the average b  across all iters.
frameTrProjOnBeta = einsum(traces_al_sm(:, ~NsExcluded, :), wNsHrLrAve, 2, 1); % (fr x u x tr) * (u x 1) --> (fr x tr)
% below is exactly same as the above. doing projection on the average
% weight is same as averaging projection. (each individually projected onto
% its svm vector)
% frameTrProjOnBeta = nanmean(einsum(traces_al_sm(:, ~NsExcluded, :), wNsHrLrNorm, 2, 1), 3); % get the projection on the svm vector of each iter and then compute the ave across iters
size(frameTrProjOnBeta) % frames x trials
frameTrProjOnBeta_hr = frameTrProjOnBeta(:, choiceVec0==1); % frames x trials % HR
frameTrProjOnBeta_lr = frameTrProjOnBeta(:, choiceVec0==0); % frames x trials % LR


av1 = nanmean(frameTrProjOnBeta_hr, 2);
av2 = nanmean(frameTrProjOnBeta_lr, 2);
mn = min([av1;av2]);
mx = max([av1;av2]);
% figure; 
subplot(221), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot(time_aligned_stimOn, av1), plot(time_aligned_stimOn, av2)
xlabel('Time since stim onset (ms)')
ylabel({'Weighted average of', 'neural responses'})
xlim([pb pe])

% frameTrProjOnBeta = einsum(traces_al_sm(:, ~NsExcluded, ~trsExcluded), SVMModel.Beta, 2, 1); % (fr x u x tr) * (u x 1) --> (fr x tr)
% size(frameTrProjOnBeta) % frames x trials
% frameTrProjOnBeta_hr = frameTrProjOnBeta(:, SVMModel.Y==1); % frames x trials % HR
% frameTrProjOnBeta_lr = frameTrProjOnBeta(:, SVMModel.Y==0); % frames x trials % LR


%% Decoder performance at each time point (decoder trained on epoch ep)

top = nanmean(avePerf, 2); % average performance across all iters.
mn = min(top(:));
mx = max(top(:));
% figure; 
subplot(222), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot(time_aligned_stimOn, top) % average across all iters
xlabel('Time since stim onset (ms)')
ylabel('Correct classification')
xlim([pb pe])


%% Weighted average of neural responses for all trials and its distribution for hr vs lr trials. U compute this on normalized x bc score=0 as the threshold for hr and lr is defined on the projection of normalized x onto the weights.
% comapre aggregate beta with beta from only 1 run of svm training.
xx = spikeAveEp0;
yy = choiceVec0; % (extraTrs); % choiceVec0

% scale xx
x = bsxfun(@minus, xx, nanmean(xx)); % (x-mu)
x = bsxfun(@rdivide, x, nanstd(xx)); % (x-mu)/sigma
x = x / SVMModel.KernelParameters.Scale; % scale is 1 unless u've changed it in svm model.


%%% Distributions
% use Beta computed from one iter
weightedAveNs_allTrs = x * SVMModel.Beta; % trials x 1

% use aggregate Beta (averaged across all iters)
weightedAveNs_allTrs = x * wNsHrLrAve; % + nanmean(biasHrLr); % score = x*beta + bias % bias should be added too but I'm not sure if averating bias across iters is the right thing to do. Also it seems like not including the bias term gives better separaton of hr and lr.


weightedAveNs_hr = weightedAveNs_allTrs(yy==1);
weightedAveNs_lr = weightedAveNs_allTrs(yy==0);
%{
weightedAveNs_allTrs = SVMModel.X * SVMModel.Beta; % trials x 1 % this is wrong u have to scale (center and normalize) SVMModel.X, bc SVMModel.Beta is for scaled data.
weightedAveNs_hr = weightedAveNs_allTrs(SVMModel.Y==1);
weightedAveNs_lr = weightedAveNs_allTrs(SVMModel.Y==0);
%}

% Compare the dist of weights for HR vs LR
[nh, eh] = histcounts(weightedAveNs_hr, 'normalization', 'probability');
[nl, el] = histcounts(weightedAveNs_lr, 'normalization', 'probability');

% figure; 
subplot(224), hold on
plot(eh(1:end-1), nh)
plot(el(1:end-1), nl)
legend('HR','LR')
xlabel('Weighted average of neurons for epoch ep')
ylabel('Fraction of trials')

% cross point of 2 gaussians when they have similar std:
mu0 = mean(weightedAveNs_hr);
mu1 = mean(weightedAveNs_lr);
sd0 = std(weightedAveNs_hr);
sd1 = std(weightedAveNs_lr);
threshes = (sd0 * mu1 + sd1 * mu0) / (sd1 + sd0);
plot([threshes threshes], [0 .5], ':')


%% Compare svm weights with random weights

popClassifierSVM_rand


%% Compare SVM weights with ROC choicePref

popClassifierSVM_choicePref







