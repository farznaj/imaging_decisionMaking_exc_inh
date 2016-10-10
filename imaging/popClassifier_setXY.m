% set X, Y, TrsExcluded, NsExcluded

frameLength = 1000/30.9; % sec.

%%
s = (stimrate-cb)'; 
allStrn = unique(abs(s));
switch strength2ana
    case 'easy'
        str2ana = (abs(s) >= (max(allStrn) - thStimStrength));
    case 'hard'
        str2ana = (abs(s) <= thStimStrength);
    case 'medium'
        str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(allStrn) - thStimStrength))); % intermediate strength
    otherwise
        str2ana = true(1, length(outcomes));
end
fprintf('Number of trials with stim strength of interest = %i\n', sum(str2ana))


%% Start setting Y: the response vector

if trialHistAnalysis
    popClassifier_trialHistory % computes choiceVec0; % trials x 1;  1 for HR choice, 0 for LR prev choice.
else
    choiceVec0 = allResp_HR_LR';  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.
    
    if strcmp(outcome2ana, 'corr')
        cprintf('blue', 'Analyzing only correct trials.\n')
        choiceVec0(outcomes~=1) = NaN; % analyze only correct trials.
        
    elseif strcmp(outcome2ana, 'incorr')
        choiceVec0(outcomes~=0) = NaN; % analyze only incorrect trials.
        cprintf('blue', 'Analyzing only incorrect trials.\n')
        
    else
        cprintf('blue', 'Analyzing both correct and incorrect trials.\n')
    end
    
    choiceVec0(~str2ana) = nan; 
    
end

cprintf('blue', '#trials for LR and HR choices = %d  %d\n', [sum(choiceVec0==0), sum(choiceVec0==1)])


%% Set ep
epStartRel2Event = ceil(ep_ms(1)/frameLength); % the start point of the epoch relative to alignedEvent for training SVM. (500ms)
epEndRel2Event = ceil(ep_ms(2)/frameLength); % the end point of the epoch relative to alignedEvent for training SVM. (700ms)
ep = eventI+epStartRel2Event : eventI+epEndRel2Event; % frames on stimAl.traces that will be used for trainning SVM.

fprintf('training epoch, rel2 stimOnset, is %.2f to %.2f ms\n', round((ep(1)-eventI)*frameLength), round((ep(end)-eventI)*frameLength))

    
%% Start setting X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.

% Compute average of spikes per frame during epoch ep.
if trialHistAnalysis
    spikeAveEp0 = squeeze(nanmean(traces_al_stimAll(ep,:,:)))';
else
    spikeAveEp0 = squeeze(nanmean(traces_al_stim(ep,:,:)))'; % trials x units.
end


% smooth the traces (moving average) using a window of size ep.
% filtered0 = boxFilter(traces_al_sm, length(ep), 1, 0);
% figure; plot(traces_al_sm(:,4,13))
% hold on, plot(filtered(:,4,13))
% spikeAveEp0(13,4)

% spikeAveEp00 = spikeAveEp0; % save it before excluding any neurons.



%% Set X Y 

%% Identify nan trials
trsExcluded = (sum(isnan(spikeAveEp0), 2) + isnan(choiceVec0)) ~= 0;


%% Exclude nan trials
X = spikeAveEp0(~trsExcluded,:); % trials x neurons
Y = choiceVec0(~trsExcluded);
fprintf('%d high-rate trials, and %d low-rate trials\n', sum(Y==1), sum(Y==0))


%% Identify neurons that are very little active.
% Little activity = neurons that are active in few trials. Also neurons that
% have little average activity during epoch ep across all trials.

% Set nonActiveNs, ie neurons whose average activity during ep is less than thAct.
spikeAveEpAveTrs = nanmean(spikeAveEp0); % 1 x units % response of each neuron averaged across epoch ep and trials.
thAct = 1e-4; %quantile(spikeAveEpAveTrs, .1);
% cprintf('m','You are using .1 quantile of average activity during ep across all neurons as threshold for identifying non-active neurons. This is arbitrary and needs evaluation!\n')
% thAct = 1e-3; % could be a good th for excluding neurons w too little activity.
nonActiveNs = spikeAveEpAveTrs < thAct;
fprintf('%d= # neurons with ave activity in ep < %.4f\n', sum(nonActiveNs), thAct)


% Set NsFewTrActiv, ie neurons that are active in very few trials (by active I mean average activity during epoch ep)
% thMinFractTrs = .05; %.01; % a neuron must be active in >= .1 fraction of trials to be used in the population analysis.
thTrsWithSpike = 1; % 3; % 1; % ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.

% nTrsWithSpike = sum(spikeAveEp0 > 0); % in how many trials each neuron
% had activity (remember this is average spike during ep).
% Remember the zero threshold used above really only makes sense if sikes
% were infered using the MCMC method, otherwise in foopsi, S has arbitrary
% values and unless a traces is all NaNs (which should not happen)
% spikeAveEp0 will be > 0 all the time.

nTrsWithSpike = sum(X > thAct); % shows for each neuron, in how many trials the activity was above thAct.
NsFewTrActiv = nTrsWithSpike < thTrsWithSpike;
fprintf('%d= # neurons that are active in less than %i trials.\n', sum(NsFewTrActiv), thTrsWithSpike)


% Now set the final NxExcluded: (neurons to exclude)
% NsExcluded = NsFewTrActiv; % remove columns corresponding to neurons with activity in <thSpTr trials.
% NsExcluded = nonActiveNs; % remove columns corresponding to neurons with <thAct activity.
NsExcluded = logical(NsFewTrActiv + nonActiveNs);

% a = size(spikeAveEp0,2) - sum(NsExcluded);
% cprintf('blue', 'included neurons= %d; total neurons= %d; fract= %.3f\n', a, size(spikeAveEp0,2), a/size(spikeAveEp0,2))
fprintf('%d = Final # non-active neurons\n', sum(NsExcluded))
fprintf('\toriginal # neurons = %d; fraction excluded = %.2f\n', size(spikeAveEp0,2), sum(NsExcluded)/size(spikeAveEp0,2))


%% Remove neurons that are very little active.
% Remove (from X) neurons that are active in few trials. Also neurons that
% have little average activity during epoch ep across all trials.

% Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
X = X(:,~NsExcluded);
% spikeAveEp0(:, NsExcluded) = []; % trials x neurons
% fprintf('# included neuros = %d, fraction = %.3f\n', size(spikeAveEp0,2), size(spikeAveEp0,2)/size(traces_al_sm,2))
% figure; plot(max(spikeAveEp))
% spikeAveEp0_sd = nanstd(spikeAveEp0);

% filtered1 = filtered0(:, ~NsExcluded, :); % includes smoothed traces only for active neurons (excluding NsExcluded).


%
% cprintf('blue', '# neurons = %d\n', size(spikeAveEp0,2))
% cprintf('blue', '# total trials = %d (half for HR and half for LR)\n', min([sum(choiceVec0==0), sum(choiceVec0==1)])*2)
numTrials = sum(~trsExcluded);
numNeurons = sum(~NsExcluded);
fprintf('%d neurons; %d trials\n', numNeurons, numTrials)

