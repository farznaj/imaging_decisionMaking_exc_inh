% compare decoding results with projecting onto a random vector (instead of
% SVMModel.Beta).

%% Set the rand vectors

nRandW = 1e4;
randWeights = randn(size(wNsHrLrAve,1), nRandW); % neurons x nrand
randWeights = bsxfun(@times, randWeights, spikeAveEp0_sd'); % stretch by std of neurons (std computed across trials).
randWeights = bsxfun(@rdivide, randWeights, sqrt(sum(randWeights.^2))); % normalize by vector length


%% see if the distribution of classifier weights are different from random neural weights
inhibNs = good_inhibit(~NsExcluded);
rs = sort(abs(randWeights), 'descend');
rs = mean(rs,2);
% figure; bar(rs)

[ws, iws] = sort(abs(wNsHrLrAve), 'descend');
wNsHrLrAveSorted = wNsHrLrAve(iws);
inhibNsSorted = inhibNs(iws);
% figure; plot(ws)
figure; 
subplot(221), hold on
bar(find(inhibNsSorted==0)-1, wNsHrLrAveSorted(~inhibNsSorted), 'b')
bar(find(inhibNsSorted==1)-1, wNsHrLrAveSorted(inhibNsSorted), 'r')

plot(rs)
plot(-rs)
xlabel('Neurons')
ylabel('Classifier weights')

[nrs, ers] = histcounts(rs, 10, 'normalization', 'probability');
[nws, ews] = histcounts(ws, 10, 'normalization', 'probability');

subplot(222), hold on
plot(ers(1:end-1), nrs)
plot(ews(1:end-1), nws)
xlabel('Classifier weight')
ylabel('Fraction of neurons')
legend('Random','Trained')

subplot(224), hold on
plot(ers(1:end-1), cumsum(nrs))
plot(ews(1:end-1), cumsum(nws))
xlabel('Classifier weight')
ylabel('Cumulative fraction of neurons')


% look at kurtosis
randKurts = kurtosis(randWeights);
wKurt = kurtosis(wNsHrLrAve);

p = 2 * mean(wKurt < randKurts);
if p > 1
  p = 2 - p;
end
% fprintf('Choice:   Kurtosis = %0.2f; median rand = %0.2f; p = %0.3f\n', ...
%   wKurt, median(randKurts), p);

subplot(223), hold on
histogram(randKurts, 'normalization', 'count')
plot([wKurt wKurt], [0 100])
xlabel('Kurtosis')
ylabel('Number of rand sims')
title(sprintf(['P = %.3f, med(rand) = %.2f'], p, median(randKurts)))



%% project the trace of all neurons on the random vectors. 

frameTrProjOnBeta = einsum(traces_al_sm(:, ~NsExcluded, :), randWeights, 2, 1);  % frames x trails x rands
size(frameTrProjOnBeta)


%% compare random projections for hr vs lr.

% r = 1;
% project spikes timeseries onto the random weights vector.
% frameTrProjOnBeta = einsum(traces_al_sm(:, ~NsExcluded, :), randWeights(:,r), 2, 1);  % frames x trails

% size(frameTrProjOnBeta) % frames x trials
frameTrProjOnBeta_hr = frameTrProjOnBeta(:, choiceVec0==1,:); % frames x trials % HR
frameTrProjOnBeta_lr = frameTrProjOnBeta(:, choiceVec0==0,:); % frames x trials % LR


av1 = nanmean(nanmean(frameTrProjOnBeta_hr, 2), 3);
av2 = nanmean(nanmean(frameTrProjOnBeta_lr, 2), 3);
sd1 = nanstd(nanmean(frameTrProjOnBeta_hr, 2), [], 3);
sd2 = nanstd(nanmean(frameTrProjOnBeta_lr, 2), [], 3);
mn = min([av1;av2]);
mx = max([av1;av2]);
figure; 
subplot(221), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot(time_aligned, av1), plot(time_aligned, av2)
% boundedline(time_aligned, av1, sd1, 'b', 'alpha')
% boundedline(time_aligned, av2, sd2, 'r', 'alpha')
xlabel('Time since stim onset (ms)')
ylabel({'Weighted average of', 'neural responses'})
xlim([pb pe])


%% compute score and labels on the random projections.

% bias should be added too to get the score... maybe just a rand number?

% to set score, x needs to be normalized.

xx = traces_al_sm(:, ~NsExcluded, :); % traces_al_sm(:, ~NsExcluded, :); % frames x units x trials
yy = choiceVec0; % (extraTrs); % choiceVec0

% scale xx
x = bsxfun(@minus, xx, nanmean(xx,2)); % (x-mu) % average across observations of each feature.
x = bsxfun(@rdivide, x, nanstd(xx,[],2)); % (x-mu)/sigma
x = x / SVMModel.KernelParameters.Scale; % scale is 1 unless u've changed it in svm model.

% compute score
s = einsum(x, randWeights, 2, 1); % + bias  % frames x trials x rands

label = s; % frames x trials x rands
label(s>0) = 1; % score<0 will be lr and score>0 will be hr, so basically threshold is 0.... 
label(s<0) = 0;
if sum(s==0)>0, error('not sure what to assign as label when score is 0!'), end

% s = frameTrProjOnBeta;
% s(frameTrProjOnBeta>0) = 1;
% s(frameTrProjOnBeta<0) = 0;

% figure; 
% subplot 221, hold on
% plot(nanmean(s(:,yy==1,45),2)) % scaled
% plot(nanmean(s(:,yy==0,45),2))
% 
% subplot 223, hold on
% ss = frameTrProjOnBeta; % non-scaled
% plot(nanmean(ss(:,yy==1,45),2))
% plot(nanmean(ss(:,yy==0,45),2))

% figure; imagesc(s(:,:,45))
% figure; imagesc(label(:,1,1))


% compute classifictaion performance
corrClass = double(bsxfun(@eq, label, yy')); % frames x trials x rands
corrClass(:, isnan(sum(s(:,:,1))) | isnan(yy'), :) = NaN; % set to nan trials that are nan either in xx or yy.
% corrClass = double(bsxfun(@(x,y)(x==0&y==0) | (x==1&y==1), label, yy')); 

% figure; imagesc(corrClass(:,:, 34))

%
avePerfRand = squeeze(nanmean(corrClass, 2));  % frames x randomIters
top = nanmean(avePerfRand,2); % frames x 1
topsd = nanstd(avePerfRand,[],2); % frames x 1
mn = min(top(:));
mx = max(top(:));
% figure; 
subplot(222), hold on
plot([0 0], [mn mx], 'k:', 'handleVisibility', 'off') % [eventI_stimOn eventI_stimOn]
plot([st st], [mn mx], 'k:', 'handleVisibility', 'off') % [epStart epStart]
plot([en en], [mn mx], 'k:', 'handleVisibility', 'off') % [epEnd epEnd]
plot(time_aligned, top) % average across all iters
boundedline(time_aligned, top, topsd, 'r', 'alpha') % average across all iters
xlabel('Time since stim onset (ms)')
ylabel('Correct classification')
xlim([pb pe])

% figure; plot(avePerfRand)


%% Weighted average of neural responses for all trials and its distribution for hr vs lr trials. U compute this on normalized x bc score=0 as the threshold for hr and lr is defined on the projection of normalized x onto the weights.
% comapre aggregate beta with beta from only 1 run of svm training.
xx = spikeAveEp0;
% a = xx;
% a(a(:)<.005) = nanmean(xx(:));
% xx = a;

yy = choiceVec0; % (extraTrs); % choiceVec0

% scale xx
x = bsxfun(@minus, xx, nanmean(xx)); % (x-mu)
x = bsxfun(@rdivide, x, nanstd(xx)); % (x-mu)/sigma
x = x / SVMModel.KernelParameters.Scale; % scale is 1 unless u've changed it in svm model.


%%% Distributions
% use Beta computed from one iter
% weightedAveNs_allTrs = x * SVMModel.Beta; % trials x 1

% use aggregate Beta (averaged across all iters)
% weightedAveNs_allTrs = x * wNsHrLrAve; % + nanmean(biasHrLr); % score = x*beta + bias % bias should be added too but I'm not sure if averating bias across iters is the right thing to do. Also it seems like not including the bias term gives better separaton of hr and lr.
s = einsum(x, randWeights, 2, 1); % + bias  % trials x rands
weightedAveNs_allTrs = nanmean(s, 2);

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


