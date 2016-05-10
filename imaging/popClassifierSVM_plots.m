% Make some summary plots for the SVM results.

% Run popClassifier to get vars needed here.


%% Plots that show the outcome of the SVM model
if doplot_svmBasics
    
    % beta and bias (intercept) terms
    figure;
    
    subplot(221)
    plot(SVMModel.Beta)
    xlabel('Neuron')
    ylabel('SVM weight')
    title(sprintf('Bias (offset) = %.3f', SVMModel.Bias))
    
    % score and posterior probabilities of each trial belonging to the positive
    % class (HR in our case).
    [label, score] = resubPredict(SVMModel);
    % [label, score] = predict(SVMModel, SVMModel.X); % does the same thing as the code above.
    [ScoreSVMModel, ScoreParameters] = fitPosterior(SVMModel); % or fitSVMPosterior
    [~, postProbs] = resubPredict(ScoreSVMModel);
    
    % score
    subplot(222)
    hold on, plot([0 size(score,1)], [0 0], 'handlevisibility', 'off')
    plot(score(:,2))
    xlabel('Trial')
    ylabel('Score') % of belonging to the positive class')
    xlim([1 SVMModel.NumObservations])
    
    % post prob
    subplot(224)
    hold on, plot([0 size(postProbs,1)], [.5 .5], 'handlevisibility', 'off')
    plot(postProbs(:,2))
    xlabel('Trial')
    ylabel('Posterior prob') % of belonging to the positive class')
    xlim([1 SVMModel.NumObservations])
    
    % label
    subplot(223), hold on
    title(['Classification error: ' num2str(sum(abs(SVMModel.Y-label))/length(label)*100) ' %'])
    plot(SVMModel.Y)
    plot(label)
    ylim([-1 2])
    xlabel('Trial')
    ylabel('Class')
    % legend('Actual', 'Model')
    xlim([1 SVMModel.NumObservations])
    
end


%% Plots related to all trials (majority of which are training dataset

figure('name', 'All trials (a large fraction is the SVM training dataset)');

pb = round((- eventI)*frameLength);
pe = round((length(time_aligned) - eventI)*frameLength);
st = round((epStart - eventI)*frameLength);
en = round((epEnd - eventI)*frameLength);


%% Simple raw averages of neural responses for HR and LR trials.
% you sould perhaps choose you training window based on this plot!

% traces = traces_al_sm(:, ~NsExcluded, :); % equal number of trs for both conds
% av1 = nanmean(nanmean(traces(:,:, choiceVec==1), 3), 2);
% av2 = nanmean(nanmean(traces(:,:, choiceVec==0), 3), 2);

traces = traces_al_sm(:, ~NsExcluded, :); % analyze all trials
av1 = nanmean(nanmean(traces(:,:, choiceVec0==1), 3), 2);
av2 = nanmean(nanmean(traces(:,:, choiceVec0==0), 3), 2);

sd1 = nanstd(nanmean(traces(:,:, choiceVec0==1), 3), 0, 2) / sqrt(size(traces, 2)); % sd of average-trial traces across neurons.
sd2 = nanstd(nanmean(traces(:,:, choiceVec0==0), 3), 0, 2) / sqrt(size(traces, 2));
% average for HR vs LR stimulus.
% av1 = nanmean(nanmean(traces(:,:, stimrate > cb), 3), 2);
% av2 = nanmean(nanmean(traces(:,:, stimrate < cb), 3), 2);

mn = min([av1;av2]);
mx = max([av1;av2]);

subplot(223), hold on
% plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI eventI]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]

boundedline(time_aligned, av1, sd1, 'b', 'alpha')
boundedline(time_aligned, av2, sd2, 'r', 'alpha')
% plot(time_aligned, av1, 'b')
% plot(time_aligned, av2, 'r')

xlabel('Time since stim onset (ms)')
ylabel('Raw average of neural responses')
xlim([pb pe])


%% Weighted average of neurons for each trial, using svm weights trained on a particular epoch (ep).

% decide what weights you want to use:
% weights = wNsHrLr; % SVMModel.Beta
weights = wNsHrLrAve; % normalized weights (and bagged in case weights
% computed from more than 1 iteration).

% include all trs (not just the random equal trs) and use the average b  across all iters.
frameTrProjOnBeta = einsum(traces_al_sm(:, ~NsExcluded, :), weights, 2, 1); % (fr x u x tr) * (u x 1) --> (fr x tr)
% below is exactly same as the above. doing projection on the average
% weight is same as averaging projection. (each individually projected onto
% its svm vector)
% frameTrProjOnBeta = nanmean(einsum(traces_al_sm(:, ~NsExcluded, :), wNsHrLrNorm, 2, 1), 3); % get the projection on the svm vector of each iter and then compute the ave across iters
% size(frameTrProjOnBeta) % frames x trials
frameTrProjOnBeta_hr = frameTrProjOnBeta(:, choiceVec0==1); % frames x trials % HR
frameTrProjOnBeta_lr = frameTrProjOnBeta(:, choiceVec0==0); % frames x trials % LR


av1 = nanmean(frameTrProjOnBeta_hr, 2);
av2 = nanmean(frameTrProjOnBeta_lr, 2);
sd1 = nanstd(frameTrProjOnBeta_hr, 0, 2) / sqrt(size(frameTrProjOnBeta_hr,2));
sd2 = nanstd(frameTrProjOnBeta_lr, 0, 2) / sqrt(size(frameTrProjOnBeta_lr,2));
mn = min([av1;av2]);
mx = max([av1;av2]);

% figure;
subplot(221), hold on
% plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI eventI]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]

boundedline(time_aligned, av1, sd1, 'b', 'alpha')
boundedline(time_aligned, av2, sd2, 'r', 'alpha')
% plot(time_aligned, av1, 'b')
% plot(time_aligned, av2, 'r')

xlabel('Time since stim onset (ms)')
ylabel({'Weighted average of', 'neural responses'})
xlim([pb pe])

% frameTrProjOnBeta = einsum(traces_al_sm(:, ~NsExcluded, ~trsExcluded), SVMModel.Beta, 2, 1); % (fr x u x tr) * (u x 1) --> (fr x tr)
% size(frameTrProjOnBeta) % frames x trials
% frameTrProjOnBeta_hr = frameTrProjOnBeta(:, SVMModel.Y==1); % frames x trials % HR
% frameTrProjOnBeta_lr = frameTrProjOnBeta(:, SVMModel.Y==0); % frames x trials % LR


%% Decoder performance at each time point (decoder trained on epoch ep)
clear hh

top = nanmean(avePerf, 2); % average performance across all iters.
mn = min(top(:));
mx = max(top(:));
% figure;
subplot(222), hold on
% plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI eventI]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot([time_aligned(1) time_aligned(end)], [.5 .5], 'k:', 'handleVisibility', 'off')
hh(1) = plot(time_aligned, top); % average across all iters
xlabel('Time since stim onset (ms)')
ylabel('Correct classification')
xlim([pb pe])


%% Shuffled data: Decoder performance at each time point (decoder trained on epoch ep)

% Set vars:
popClassifierSVM_traindata_set_plot

% plot
subplot(222), hold on
id = 2;
top = nanmean(avePerf_cv_dataVSshuff{id}, 2); % average performance across all iters.
top_sd = nanstd(avePerf_cv_dataVSshuff{id}, 0, 2) / sqrt(size(avePerf_cv_dataVSshuff{id},2));

mn = min(top(:));
mx = max(top(:));
% mxa = [mxa, mx];


% % plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI eventI]
% plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
% plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
% plot([time_aligned(1) time_aligned(end)], [.5 .5], 'k:', 'handleVisibility', 'off')

% plot(time_aligned, top) % average across all iters
hh(id) = boundedline(time_aligned, top, top_sd, col{id}, 'alpha');

xlabel('Time since stim onset (ms)')
ylabel('Correct classification')
xlim([pb pe])



%%% mark frames with significant different in class accuracy between
%%% shuffled and actual data
p_classAcc = NaN(1, size(avePerf_cv,1));
h_classAcc = NaN(1, size(avePerf_cv,1));
for fr = 1:length(p_classAcc)
    [h_classAcc(fr), p_classAcc(fr)]= ttest2(avePerf, avePerf_cv_dataVSshuff{2}(fr,:),...
        'tail', 'right', 'vartype', 'unequal'); % Test the alternative hypothesis that the population mean of actual data is less than the population mean of shuffled data.
end


hnew = h_classAcc;
hnew(~hnew) = nan;
plot(time_aligned, hnew * mn -.02, 'k')

legend(hh, {'data', 'shuffled'}, 'location', 'northwest')
legend boxoff


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
weightedAveNs_allTrs = x * weights; % SVMModel.Beta; % trials x 1
% use aggregate Beta (averaged across all iters)
% weightedAveNs_allTrs = x * wNsHrLrAve; % + nanmean(biasHrLr); % score = x*beta + bias % bias should be added too but I'm not sure if averating bias across iters is the right thing to do. Also it seems like not including the bias term gives better separaton of hr and lr.


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

