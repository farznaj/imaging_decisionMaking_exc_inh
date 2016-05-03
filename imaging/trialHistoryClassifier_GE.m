doplots = true;
binRates = true;
doiti = true;
binITIs = true; % false; %true;
vec_iti = [0 10 30]; %[0 6 9 12 30]; % [0 7 30]; % [0 10 30]; % [0 6 9 12 30]; % use [0 40]; if you want to have a single iti bin and in conventioinal analysis look at the effect of current rate on outcome.
vec_ratesdiff = 0:2:12;
defaultHelpedTrs = false; % false; % set to 0 if you want to manually set the helped trials.
saveHelpedTrs = true;
allowCorrectResp = 'change';
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';

excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;
th = 5; % 10; 9; % this number of beginning trials will be excluded. also later in the code we want >th trials in each column of ratesDiffInput

%% set outcome and response side for each trial, taking into account allcorrection and uncommitted responses.

[outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp);
% Remember alldata.outcome is not necessarily the same as outcomes,
% depending on what you do to allCorrection trials, if it is set to change,
% then outcomes and alldata.outcome will be opposite to each other!!!
%% set stim rate

[~, ~, stimrate, stimtype] = setStimRateType(alldata); % stimtype = [multisens, onlyvis, onlyaud];
cb = unique([alldata.categoryBoundaryHz]); % categ boundary in hz

%% set the response (choice) vector: 0 for LR and 1 for HR.
% y(tr)=1 if tr was a HR choice, y(tr)=0 if tr was a LR choice, y(tr)=nan, if tr was a trs2rmv or had an outcome other than success, failure.

y = allResp_HR_LR';
y(trs2rmv) = NaN;
y(isnan(stimrate)) = NaN;

% y(stimtype(:,1)~=1) = nan;

good_corr_incorr = ~isnan(y'); % trials that are not among trs2rmv and they ended up being either correct or incorrect. (doesn't include trials with any other outcome.)
% num_final_trs = sum(good_corr_incorr), fract_final_trs = mean(good_corr_incorr) % number and fraction of trials that will be used in the model fit.

if sum(good_corr_incorr)<20
    warning('Too few trials remained after applying trs2rmv! aborting...')
    B = []; deviance = []; stats = [];
    X = []; y = []; vec_ratesdiff = []; ratesDiffVecInds = []; vec_iti = []; itiVecInds = [];
    ratesDiffInput = []; successPrevInput = []; failurePrevInput = []; itiPrecedInput = [];
    return
    
elseif doplots
    figh = figure('name', ['Mouse ', mouse]);    
end



%% set bzero (constant) vector
bzeroInput = ones(length(alldata), 1);
%% set vars for the stim rate matrix.

stimrate(isnan(y)) = nan;

if ~binRates
    ratesDiffInput = stimrate-cb;
    ratesDiffInput = ratesDiffInput / max(abs(ratesDiffInput)); % normalize it so it is in the same range as other input vectors (success, failure, etc).
    
else
    if ~exist('vec_ratesdiff', 'var') || isempty(vec_ratesdiff)
        wd = 2;
        vec_ratesdiff = 0: wd: max(stimrate)+wd-cb; % 0: wd: max(abs(stimrate-cb))+wd
        % vec_rates = sort([cb : -wd : min(stimrate)-wd  ,  cb+wd : wd : max(stimrate)+wd]); % make sure categ boundary doesn't go at the middle of a bin!
        % [n, bin_rate] = histc(stimrate, vec_rates);
    else
        wd = unique(diff(vec_ratesdiff));
    end
    
    [~,~, bin_ratediff] = histcounts(abs(stimrate-cb), vec_ratesdiff);
    ratesDiffVecInds = unique(bin_ratediff);
    ratesDiffVecInds(~ratesDiffVecInds) = []; % ind 0 corresponds to nans, so exclude it.
    
    
    %%%% set (stimrate - cb) input matrix (for current trial)
    
    % ratesDiffInput = zeros(length(alldata), length(ratesDiffVecInds));
    ratesDiffInput = NaN(length(alldata), length(ratesDiffVecInds));
    
    % for itr = 1:length(alldata)
    for itr = find(good_corr_incorr)    % set ratesDiffInput only for trials are not tr2rmv or that the outcome was 0 or 1.
        ratesDiffInput(itr,:) = ismember(ratesDiffVecInds, bin_ratediff(itr)); % for each trial only one column will be non-zero, and that column index is the bin index of the trial's abs(stimrate-cb).
        ratesDiffInput(itr,:) = sign(stimrate(itr)-cb) * ratesDiffInput(itr,:); % assign 1 if the trial was HR and -1 if it was LR.
    end
    % If ratesDiffInput(tr,rv)=1 or -1, then trial tr's abs(stimrate-cb) was in
    % bin rv of vec_ratesdiff, ie within [vec_ratesdiff(rv) vec_ratesdiff(rv+1)-eps]
    %
    % If ratesDiffInput(tr,rv)=0, then trial tr's abs(stimrate-cb) was NOT in
    % bin rv of vec_ratesdiff, EXCEPT for when stimrate(tr)=cb. In this case
    % ratesDiffInput(tr,rv) should be 1 or -1 but bc you multiply it by
    % sign(stimrate(itr)-cb), it gets 0. So 0s in ratesDiffInput(:,1) are
    % either bc the trial's abs(stimrate-cb) was in a bin other than rv, or bc
    % its abs(stimrate-cb) was 0.
    
    % figure; imagesc(ratesDiffInput)
    
end



%% set success vector for the previous trial

% Note regarding how prev sucess and failure vectors are set:
% Remember for the explanation below, you need to look at outcomes and not
% alldata.outcome (they could be different if you set allowCorrectResp to
% change). See above for more detialed explanation.
% successPrevInput(i) indicates the success outcome and choice of trial i-1:
% if successPrevInput(i)=1, then trial i-1 was a successful HR-choice
% trial.
% if successPrevInput(i)=-1, then trial i-1 was a successful LR-choice
% trial.
% if successPrevInput(i)=0, then trial i-1 was a failure trial.
%
% failurePrevInput(i) indicates the failure outcome and choice of trial i-1:
% if failurePrevInput(i)=1, then trial i-1 was a failure HR-choice
% trial.
% if failurePrevInput(i)=-1, then trial i-1 was a failure LR-choice
% trial.
% if failurePrevInput(i)=0, then trial i-1 was a successful trial.
%


% set s_curr indicating the success/failure and HR/LR outcome of the
% current trial.
ynew = y;
ynew(y==0) = -1; % in ynew LR will be -1 and HR will remain 1.
s_curr = ynew .* (outcomes==1)'; % 0 or nan for any output other than success. 1 for succ HR. -1 for succ LR.

% set s_prev indicating the success/failure and HR/LR outcome of the
% previous trial. shift s_curr one element front.
successPrevInput = [NaN; s_curr(1:end-1)]; % s_prev(i) = s_curr(i-1);
successPrevInput(begTrs) = NaN; % no previous trial exists for the 1st trial of a session.

a = find(~good_corr_incorr); % these trials will be nan in y and ratesDiffInput
b = find(isnan(successPrevInput)); % these trials will be nan in successPrevInput and failurePrevInput
nan_y_s_f = union(a,b); % trials that are nan either in y or succ/fail vectors.
% num_fitted_trs = length(y) - length(nan_y_s_f) % final number of trials that will be using in the fitting.


%% set failure vector for the previous trial

% set f_curr indicating the success/failure and HR/LR outcome of the
% current trial.
f_curr = ynew .* (outcomes==0)'; % 0 or nan for any output other than failure. 1 for fail HR. -1 for fail LR.

% set f_prev indicating the success/failure and HR/LR outcome of the
% previous trial. shift f_curr one element front.
failurePrevInput = [NaN; f_curr(1:end-1)]; % s_prev(i) = s_curr(i-1);
failurePrevInput(begTrs) = NaN; % no previous trial exists for the 1st trial of a session.

%% take care of ITI input

if doiti
    %% compute ITI
    % many ways to define it. for now: iti = time between commiting a choice in
    % trial i and stimulus onset in trial i+1.
    % if no committed choice or no stim onset set iti to nan.
    
    % how about iti = stim onset to stim onset? you can plot the two.
    
    % time of making choice
    t0 = NaN(length(alldata), 1);
    if strcmp(uncommittedResp, 'change')
        % use time of 1st lick (not necessarily got committed but you defined outcome based on it.).
        for itr = 1:length(alldata)
            if outcomes(itr)==1
                t0(itr) = alldata(itr).parsedEvents.states.correctlick_again_wait(1);
                
            elseif outcomes(itr)==0
                t0(itr) = alldata(itr).parsedEvents.states.errorlick_again_wait(1);
            end
        end
        
    else % use time of commit response
        for itr = 1:length(alldata)
            if outcomes(itr)==1
                t0(itr) = alldata(itr).parsedEvents.states.reward(1);
                
            elseif outcomes(itr)==0
                if ~isempty(alldata(itr).parsedEvents.states.punish)
                    t0(itr) = alldata(itr).parsedEvents.states.punish(1);
                    
                else % punish allow correction entered.
                    t0(itr) = alldata(itr).parsedEvents.states.punish_allowcorrection(1);
                    %                     pac = alldata(itr).parsedEvents.states.punish_allowcorrection;
                    %                     pacd = alldata(itr).parsedEvents.states.punish_allowcorrection_done;
                    %                     t0(itr) = pac(ismember(pac(:,2), pacd(1)),1);       % this is not needed. mouse commits his wrong choice once he enters the pucnish_allcorrection state.
                end
            end
        end
    end    
    
    % time of stim onset
    t1 = NaN(length(alldata), 1);
    for itr = 1:length(alldata)
        if ~isempty(alldata(itr).parsedEvents.states.wait_stim) % ~ismember(outcomes(itr), -3) % wrong init.
            t1(itr) = alldata(itr).parsedEvents.states.wait_stim(1);
        end
    end    
    
    % itiProced will be NaN if a trial was wrong init or if its preceding trial
    % did not have a committed choice.
    itiPreced = NaN(length(alldata), 1);
    itiPreced(2:end) = t1(2:end) - t0(1:end-1); % in sec, stimOn of current trial - choiceCommit of previous trial
    itiPreced(begTrs) = NaN; % no previous trial exists for the 1st trial of a session.       
%     min_max_iti = [min(itiPreced)  max(itiPreced)]


    itiPreced(nan_y_s_f) = NaN; % you don't need to do this bc these trials wont be anyway fitted (they're nan in y or s or f), but u do it to make the iti bins fewer.
    fprintf('ITI, min: %.2f, max: %.2f\n', min(itiPreced), max(itiPreced))
    
    if doplots
        figure(figh); subplot(421),
        histogram(itiPreced)
        xlabel('ITI (sec)')
        ylabel('number of trials')
    end
    
%     warning('define the outlier value for iti... using arbitrary values...')
    %     itiPreced(itiPreced > 50) = NaN; % 3*quantile(itiPreced, .9)
    itiPreced(itiPreced > 30) = NaN; % 3*quantile(itiPreced, .75)
    %     itiPreced(itiPreced > 15) = NaN; % 3*nanstd(itiPreced)
    
    c = find(isnan(itiPreced));
    nan_y_s_f_t = union(nan_y_s_f, c); % trials that are nan either in y or succ/fail vectors.
    
    %% set ITI matrix
    
    if ~binITIs % go with the actual values of ITI.
        itiPrecedN = itiPreced / max(itiPreced); % normalize it so it is in the same range as other input vectors (success, failure, etc).
        itiSuccPrecedInput = itiPrecedN .* successPrevInput;
        itiFailPrecedInput = itiPrecedN .* failurePrevInput;
        
    else % bin the ITIs
        if ~exist('vec_iti', 'var') || isempty(vec_iti)
            %         vec_iti = floor(min(itiPreced)) : ceil(max(itiPreced));
            %         vec_iti = [min(itiPreced)-1 6 9 12 max(itiPreced)+1];
            %         vec_iti = [min(itiPreced)-1 6 9 max(itiPreced)+1];
            %         vec_iti = [min(itiPreced)-1 7 9 max(itiPreced)+1];
            vec_iti = [min(itiPreced)-1 7 max(itiPreced)+1];
            %         vec_iti = [min(itiPreced)-1 10 max(itiPreced)+1];
            %         vec_iti = [min(itiPreced)-1 max(itiPreced)+1];
        end
        
        
        %%
        [~, bin_iti] = histc(itiPreced, vec_iti);
        itiVecInds = unique(bin_iti);
        itiVecInds(~itiVecInds) = []; % bin_iti=0 corresponds to NaN values of itiPreced. exclude them.
        
        
        itiPrecedInput = NaN(length(alldata), length(itiVecInds));
        for itr = 1:length(alldata)
            itiPrecedInput(itr,:) = ismember(itiVecInds, bin_iti(itr)); % for each trial only one column will be non-zero, and that column index is the bin index of the trial's iti.
        end
        itiPrecedInput(isnan(itiPreced),:) = NaN; % try not doing this and setting all columns to 0 for these trials.
        
        %         itiHRLRPrecedInput = bsxfun(@times, itiPrecedInput, successPrevInput+failurePrevInput);
        % figure; imagesc(itiHRLRPrecedInput)
        % (nansum(abs(itiHRLRPrecedInput)))
        
        itiSuccPrecedInput = bsxfun(@times, itiPrecedInput, successPrevInput);
        % figure; imagesc(itiSuccPrecedInput)
        
        itiFailPrecedInput = bsxfun(@times, itiPrecedInput, failurePrevInput);
        % figure; imagesc(itiFailPrecedInput)
        
        
        %% number of trials that went into each bin for iti_succ and iti_fail
        
        if doplots
            figure(figh); subplot(422), hold on
            plot(vec_iti(itiVecInds), nansum(abs(itiSuccPrecedInput)), 'o-')
            plot(vec_iti(itiVecInds), nansum(abs(itiFailPrecedInput)), 'o-')
            xlabel('ITI (sec)')
            ylabel('number of trials')
        end        
        
    end    
else
    nan_y_s_f_t = nan_y_s_f;
end
   

%% set number of trials that will go into the model.

num_fitted_trs = length(y) - length(nan_y_s_f_t); % final number of trials that will be using in the fitting.
fprintf('Number of trials, original: %d, fitted: %d\n', length(y), num_fitted_trs)

%% now that all nan trials are determined, make sure that there is no column of ratesDiffInput with <=th trials.

if binRates
    sr = stimrate(~isnan(y));
    n = histcounts(abs(sr-cb), vec_ratesdiff);
%     th = 9; % we want >th trials in each column of ratesDiffInput
    ratesDiffInput = ratesDiffInput(:, ismember(ratesDiffVecInds, find(n>th))); % only keep those columns of ratesDiffInput that have more than th trials.
    ratesDiffVecInds = find(n>th);
else
    vec_ratesdiff = []; 
    ratesDiffVecInds = [];
end

if ~binITIs
    vec_iti = []; 
    itiVecInds = [];
end


% set some params related to behavior and trial types.
% behavior_info

%% specify what traces you want to plot
traces = alldataSpikesGood; % alldataSpikesGoodInh; % %  % traces to be aligned.


%%
traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.

defaultPrePostFrames = 2; % default value for nPre and postFrames, if their computed values < 1.
shiftTime = 0; % event of interest will be centered on time 0 which corresponds to interval [-frameLength/2  +frameLength/2]
scaleTime = frameLength;

alignedEvent = 'initTone';
[traces_aligned_fut_initTone, time_aligned_initTone, eventI_initTone] = alignTraces_prePost_allCases...
    (alignedEvent, traces, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, trs2rmv);



%% Align traces on particular trial events


% remember traces_al_sm has nan for trs2rmv as well as trs in alignedEvent that are nan.

traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
alignedEvent = 'initTone'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
dofilter = false; true;

traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.

[traces_al_sm, time_aligned_stimOn, eventI_stimOn] = alignTraces_prePost_filt(traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward);

% set to nan those trials in outcomes and allRes that are nan in traces_al_sm
a = find(sum(sum(~isnan(traces_al_sm),1),3), 1);
allTrs2rmv = find(squeeze(sum(isnan(traces_al_sm(:,a,:)))));
outcomes(allTrs2rmv) = NaN; 
allResp(allTrs2rmv) = NaN; 

%% the average of frames prior to initian tone.
traces_al_sm_aveFr = nanmean(traces_al_sm(1: eventI_stimOn,:,:), 1);












%%
%%
%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Remember: you can use the script svmUnderstandIt to understand how some
% of the matlab functions related to SVM classification work.
prevSuccessFlg = false; % true previous sucess trials; false: previous failure.
iTiFlg = 2; % 0: short iTi, 1: long iTi, 2: all iTis.
windowAvgFlg = true;
pcaFlg = true;


thAct = 1e-3; % could be a good th for excluding neurons w too little activity.

doplots = true;
epStart = 1;
epEnd = eventI_stimOn;

rng(0, 'twister'); % Set random number generation for reproducibility


%% Set Y: the response vector (need work here)

switch iTiFlg
    case 0
        if prevSuccessFlg
            allResp_HR_LR = itiSuccPrecedInput(:,1);
            allResp_HR_LR(allTrs2rmv) = NaN; % remove the nans from the data
            allResp_HR_LR(allResp_HR_LR==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR(allResp_HR_LR<0) = 0; % change the code for LR from -1 to 0
        else
            allResp_HR_LR = itiFailPrecedInput(:,1);
            allResp_HR_LR(allTrs2rmv) = NaN; % remove the nans from the data
            allResp_HR_LR(allResp_HR_LR==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR(allResp_HR_LR<0) = 0; % change the code for LR from -1 to 0
        end
        
    case 1
        if prevSuccessFlg
            allResp_HR_LR = itiSuccPrecedInput(:,2);
            allResp_HR_LR(allTrs2rmv) = NaN; % remove the nans from the data
            allResp_HR_LR(allResp_HR_LR==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR(allResp_HR_LR<0) = 0; % change the code for LR from -1 to 0
        else
            allResp_HR_LR = itiFailPrecedInput(:,2);
            allResp_HR_LR(allTrs2rmv) = NaN; % remove the nans from the data
            allResp_HR_LR(allResp_HR_LR==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR(allResp_HR_LR<0) = 0; % change the code for LR from -1 to 0
        end
        
    case 2
        if prevSuccessFlg
            allResp_HR_LR = successPrevInput;
            allResp_HR_LR(allTrs2rmv) = NaN; % remove the nans from the data
            allResp_HR_LR(allResp_HR_LR==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR(allResp_HR_LR<0) = 0; % change the code for LR from -1 to 0
        else
            allResp_HR_LR = failurePrevInput;
            allResp_HR_LR(allTrs2rmv) = NaN; % remove the nans from the data
            allResp_HR_LR(allResp_HR_LR==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR(allResp_HR_LR<0) = 0; % change the code for LR from -1 to 0
        end
end
choiceVec0 = allResp_HR_LR(:);  % trials x 1;  1 for HR choice, 0 for LR prev choice.
fprintf('N trials for LR and HR = %d  %d\n', [sum(choiceVec0==0), sum(choiceVec0==1)])

%% Set X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.

% Compute average of spikes per frame during epoch ep.
spikeAveEp0 = squeeze(traces_al_sm_aveFr)'; % trials x units.


%% Remove (from X) neurons with average activity during epoch ep in too few trials. or with too little average activity during epoch ep and averaged across all trials.
%
thMinFractTrs = .05; %.01; % a neuron must be active in >= .1 fraction of trials to be used in the population analysis.
thTrsWithSpike = 3; ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.
nTrsWithSpike = sum(spikeAveEp0 > 0); % in how many trials each neuron had activity (remember this is average spike during ep).
NsFewTrActiv = nTrsWithSpike < thTrsWithSpike;


spikeAveEpAveTrs = nanmean(spikeAveEp0); % 1 x units % response of each neuron averaged across epoch ep and trials.
% thAct = 1e-3; % could be a good th for excluding neurons w too little activity.
nonActiveNs = spikeAveEpAveTrs < thAct;
fprintf('Number of non-active Ns= %d \n', sum(nonActiveNs))

% NsExcluded = NsFewTrActiv; % remove columns corresponding to neurons with activity in <thSpTr trials.
% NsExcluded = nonActiveNs; % remove columns corresponding to neurons with <thAct activity.
NsExcluded = logical(NsFewTrActiv + nonActiveNs);

spikeAveEp0(:, NsExcluded) = [];
fprintf('# included neuros = %d, fraction = %.3f\n', size(spikeAveEp0,2), size(spikeAveEp0,2)/size(traces_al_sm,2))
% figure; plot(max(spikeAveEp))
spikeAveEp0_sd = nanstd(spikeAveEp0);

%%
wNsHrLr = NaN(size(spikeAveEp0,2), 1);
biasHrLr = NaN(1, 1);
fractMisMatch = NaN(1, 1);
avePerf = NaN(size(traces_al_sm,1), 1);

%% Use equal number of trials for both HR and LR conditions.

extraTrs = setRandExtraTrs(find(choiceVec0==0), find(choiceVec0==1)); % find extra trials of the condition with more trials, so u can exclude them later.

choiceVec = choiceVec0;
% make sure choiceVec has equal number of trials for both lr and hr.
choiceVec(extraTrs) = NaN; % set to nan some trials (randomly chosen) of the condition with more trials so both conditions have the same number of trials.
trsExcluded = isnan(choiceVec);

fprintf('N trials for LR and HR = %d  %d\n', [sum(choiceVec==0), sum(choiceVec==1)])



% Make sure spikeAveEp has equal number of trials for both lr and hr.
spikeAveEp = spikeAveEp0;
spikeAveEp(extraTrs,:) = NaN; % set to nan some trials (randomly chosen) of the condition with more trials so both conditions have the same number of trials.
%%
xx = spikeAveEp0; % spikeAveEp0(extraTrs,:); % trials x units
yy = choiceVec0; % choiceVec0(extraTrs); % trials x 1

mskNan = isnan(choiceVec);
if windowAvgFlg
    X = spikeAveEp(~mskNan, :); % spikeAveEp0(extraTrs,:); % trials x units
    Y = choiceVec(~mskNan); % choiceVec0(extraTrs); % trials x 1
else
    X = reshape(permute(traces_al_sm(ep, ~NsExcluded, ~mskNan), [1 3 2]),...
    length(ep)*sum(~mskNan), sum(~NsExcluded));
    Y = repmat(reshape(choiceVec(~mskNan), 1, sum(~mskNan)), length(ep), 1);
    Y = Y(:);  
end

%% run SVM
cnam = [0,1]; % LR: negative ; HR: positive
% SVMModel = svmClassifierMS(X, Y, cnam);
if pcaFlg 
    [PCs, ~, l] = pca(X);
    numPCs = find(cumsum(l/sum(l))>0.99, 1, 'first');
    X_s = bsxfun(@plus, bsxfun(@minus, X, mean(X))*(PCs(:, 1:numPCs)*PCs(:, 1:numPCs)'), mean(X));
    SVMModel = fitcsvm(X_s, Y, 'standardize', 1, 'ClassNames', cnam, 'KernelFunction', 'linear'); % 'KernelFunction'. 'BoxConstraint'
else
    SVMModel = fitcsvm(X, Y, 'standardize', 1, 'ClassNames', cnam, 'KernelFunction', 'linear'); % 'KernelFunction'. 'BoxConstraint'
end
wNsHrLr(:,1) = SVMModel.Beta;
biasHrLr(1) = SVMModel.Bias;
            
fprintf('# neurons = %d\n', size(SVMModel.Mu, 2))
fprintf('# total trials = %d\n', SVMModel.NumObservations)
fprintf('# trials that are support vectors = %d\n', size(SVMModel.Alpha,1))

CVSVMModel = crossval(SVMModel);
CVSVMModel.kfoldLoss

%% compute label for all trs (not just the equal number trs that were
% randomly selected) and see how well it matches with the actual class.    
[label] = predict(SVMModel, xx); % predict(SVMModel, SVMModel.X);    
label(isnan(sum(xx,2))) = NaN;
fractMisMatch(1) = sum(abs(yy - label)>0) / sum(~isnan(yy - label));

%% see how well the SVM trained on our particular epoch can decode other time points.

corrClass = NaN(size(traces_al_sm,1), size(traces_al_sm,3)); % frames x trials

for itr = 1 : size(traces_al_sm,3)
    % u may wanna smooth traces_al_sm(:, ~NsExcluded, itr)
    a = traces_al_sm(:, ~NsExcluded, itr); % frames x neurons
    if any(isnan(a(:)))
        if ~all(isnan(a(:))), error('how did it happen?'), end
    elseif ~isnan(choiceVec0(itr))
        l = predict(SVMModel, a);
        corrClass(:, itr) = (l==choiceVec0(itr));
    end
end
% average performance (correct classification) across trials.
avePerf(:,1) = nanmean(corrClass, 2);  % frames x randomIters
    %% Make some plots to evaluate the SVM model
    if ~SVMModel.ConvergenceInfo.Converged, error('not converged!'), end
    
        fprintf('converged = %d\n', SVMModel.ConvergenceInfo.Converged)
        % SVMModel.NumObservations == size(choiceVec,1) - (length(extraTrs) + sum(isnan(choiceVec0))) % final number of trials
        % size(SVMModel.X,2) == size(spikeAveEp0,2) - sum(NsFewTrActiv) % final number of neurons
        
        if any(SVMModel.Prior ~= .5), error('The 2 conditions have non-equal number of trials!'), end
        %     fprintf('Prior probs = %.3f  %.3f\n', SVMModel.Prior) % should be .5 for both classes unless you used different number of trials for each class.
        
        % beta and bias (intercept) terms
        figure;
        
        subplot(221)
        plot(SVMModel.Beta)
        xlabel('Neuron')
        ylabel('SVM weight')
        title(sprintf('Bias = %.3f', SVMModel.Bias))
        
        % score and posterior probabilities of each trial belonging to the positive
        % class (HR in our case).
        [label, score] = resubPredict(SVMModel);
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
%%  quality relative to shuffles
       
classLossTrain = [];
classLossTest = [];
classLossChanceTrain = [];
classLossChanceTest = [];

wNsHrLr_s = [];
biasHrLr_s = [];
wNsHrLrChance = [];
biasHrLrChance = [];

for s = 1:100
    shflTrials = randperm(length(Y));
    X_s = X(shflTrials, :);
    Y_s = Y(shflTrials);
%%%%%%%% reduce features by PCA
    if pcaFlg
        X_s = bsxfun(@plus, bsxfun(@minus, X_s, mean(X_s))*(PCs(:, 1:numPCs)*PCs(:, 1:numPCs)'), mean(X_s));
    end
     
%%%%%%%% data augmentation resampling
% % % %             mskNans = ~isnan(choiceVec);
% % % %             X_s = spikeAveEp(mskNans, :);
% % % %             Y_s = choiceVec(mskNans);
% % % %             numNeus = size(X_s, 2);
% % % %             X_sss = [];
% % % %             Y_sss = [];
% % % %             for ss = 1:3
% % % %                msk1 = Y_s == 1;
% % % %                X_s1 = X_s(msk1, :);
% % % %                Y_s1 = Y_s(msk1, :);
% % % %                X_s0 = X_s(~msk1, :);
% % % %                Y_s0 = Y_s(~msk1, :);
% % % %                X_ss = [];
% % % %                for n = 1:numNeus
% % % %                    X_ss(1:length(Y_s1), n) = X_s1(randi(length(Y_s1), length(Y_s1), 1), n);
% % % %                    X_ss(length(Y_s1)+(1:length(Y_s0)), n) = X_s0(randi(length(Y_s0), length(Y_s0), 1), n);
% % % %            
% % % %                end
% % % %                X_sss = [X_sss; X_ss];
% % % %                Y_sss = [Y_sss; Y_s1; Y_s0];
% % % %             end
% % % %             X_s = X_sss;
% % % %             Y_s = Y_sss;
%%%%%%%%%   


    SVMModel_s = fitcsvm(X_s, Y_s, 'standardize', 1, 'ClassNames', cnam); % Linear Kernel            
    classLossTrain(s) = mean(abs(Y_s-predict(SVMModel_s, X_s)));
    wNsHrLr_s(:, s) = SVMModel_s.Beta;
    biasHrLr_s(:, s) = SVMModel_s.Bias;
    
    CVSVMModel_s = crossval(SVMModel_s, 'kfold', 10); % CVSVMModel.Trained{1}: model 1 --> there will be KFold of these models. (by default KFold=10);
    classLossTest(s) = kfoldLoss(CVSVMModel_s); % Classification loss (by default the fraction of misclassified data) for observations not used for training
    Y_s_shfld = Y_s(randperm(length(Y_s)));
    SVMModelChance = fitcsvm(X_s, Y_s_shfld, 'standardize', 1, 'ClassNames', cnam); %  % Linear Kernel
    CVSVMModelChance = crossval(SVMModelChance, 'kfold', 10); % CVSVMModel.Trained{1}: model 1 --> there will be KFold of these models. (by default KFold=10);
    wNsHrLrChance(:, s) = SVMModelChance.Beta;
    biasHrLrChance(:, s) = SVMModelChance.Bias;
    
    classLossChanceTrain(s) = mean(abs(Y_s_shfld-predict(SVMModelChance, X_s)));
    classLossChanceTest(s) = kfoldLoss(CVSVMModelChance); % Classification loss (by default the fraction of misclassified data) for observations not used for training
    s
end
classLoss = mean(classLossTest);
fprintf('Average cross-validated classification error = %.3f\n', (classLoss))
figure;
subplot(211)
hold on
%         hd = hist(classLossTrain, 0:0.02:1);
hc = hist(classLossChanceTrain, 0:0.02:1);
bar(0:0.02:1, hc, 'facecolor', 0.5*[1 1 1], 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
%         bar(0:0.02:1, hd, 'facecolor', 'r', 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
plot(mean(classLossTrain), 0, 'ko','markerfacecolor', 'r', 'markersize', 6)
plot(mean(classLossChanceTrain), 0, 'ko','markerfacecolor', 0.5*[1 1 1], 'markersize', 6)
ylabel('Count')
xlabel('training loss')
xlim([0 1])
legend('shuffled', 'data', 'location', 'northwest')
legend boxoff

subplot(212)
[~, p]= ttest2(classLossTest, classLossChanceTest, 'tail', 'left', 'vartype', 'unequal');
hold on
hd = hist(classLossTest, 0:0.02:1);
hc = hist(classLossChanceTest, 0:0.02:1);
bar(0:0.02:1, hc, 'facecolor', 0.5*[1 1 1], 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
bar(0:0.02:1, hd, 'facecolor', 'r', 'edgecolor', 'none', 'Facealpha', 0.7', 'barwidth', 1);
plot(mean(classLossTest), 0, 'ko','markerfacecolor', 'r', 'markersize', 6)
plot(mean(classLossChanceTest), 0, 'ko','markerfacecolor', 0.5*[1 1 1], 'markersize', 6)
ylabel('Count')
title(['p-value lower tail = ' num2str(p)])
xlabel('cross-validation loss')
xlim([0 1])
legend('shuffled', 'data', 'location', 'northwest')
legend boxoff
  
fprintf('Average cross-validated classification error = %.3f\n', mean(classLoss))
%         cl(i) = classLoss;
%         end
% Estimate cross-validation predicted labels and scores.
% For every fold, kfoldPredict predicts class labels for in-fold
% observations using a model trained on out-of-fold observations.
[elabel, escore] = kfoldPredict(CVSVMModel);

% Estimate the out-of-sample posterior probabilities
[ScoreCVSVMModel, ScoreParameters] = fitSVMPosterior(CVSVMModel);
[~, epostp] = kfoldPredict(ScoreCVSVMModel);
% How claassLoss is computed? I think: classLoss = 1 - mean(label == elabel)
diff([classLoss, mean(label ~= elabel)])


%% Average b across all iters (bagging : bootstrap aggregation)

% figure; imagesc(wNsHrLr)
% figure; errorbar(1:size(wNsHrLr,1), mean(wNsHrLr, 2), std(wNsHrLr, [], 2), 'k.')

bLen = sqrt(sum(wNsHrLr.^2)); % norm of wNsHrLr for each rand
% figure; plot(bLen)
wNsHrLrNorm = bsxfun(@rdivide, wNsHrLr, bLen); % normalize b of each rand by its vector length 
wNsHrLrAve = mean(wNsHrLrNorm, 2); % average of normalized b across all rands.
wNsHrLrAve = wNsHrLrAve / norm(wNsHrLrAve); % normalize it so the final average vector has norm of 1.
% figure; plot(bNsHrLrAve)
% figure; errorbar(1:size(wNsHrLr,1), mean(wNsHrLrNorm, 2), std(wNsHrLrNorm, [], 2), 'k.')



%% compute fraction of mismatch in classification

xx = spikeAveEp0;
yy = choiceVec0; % (extraTrs); % choiceVec0

% scale xx
x = bsxfun(@minus, xx, nanmean(xx)); % (x-mu)
x = bsxfun(@rdivide, x, nanstd(xx)); % (x-mu)/sigma
x = x / SVMModel.KernelParameters.Scale; % scale is 1 unless u've changed it in svm model.


% compute score on the aggregate of all iters of svm.
s = x * wNsHrLrAve + nanmean(biasHrLr); % score = x*beta + bias % bias should be added too but I'm not sure if averating bias across iters is the right thing to do. Also it seems like not including the bias term gives better separaton of hr and lr.

label = s;
label(s>0) = 1; % score<0 will be lr and score>0 will be hr, so basically threshold is 0.... 
label(s<0) = 0;
if sum(s==0)>0, error('not sure what to assign as label when score is 0!'), end

fractMisMatchFinal = sum(abs(yy - label)>0) / sum(~isnan(yy - label));

[nanmean(fractMisMatch) fractMisMatchFinal]


% compare with fractMisMatch on each iter ... see if doing several
% iters helped w better prediction:
figure; hold on,
plot([0 length(fractMisMatch)], [fractMisMatchFinal fractMisMatchFinal])
plot(fractMisMatch)
title(sprintf('%.3f  %.3f', nanmean(fractMisMatch), fractMisMatchFinal))


%{
%% plot bias
figure('name', 'bias term'); subplot(211), plot(biasHrLr)
subplot(212),  errorbar( mean(biasHrLr), std(biasHrLr), 'k.')
%}


%%
%% Plots
%%


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
xlabel('Time since initiation tone onset (ms)')
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
xlabel('Time since initiation tone onset (ms)')
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
xlabel('Time since initiation tone onset (ms)')
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

% popClassifierSVM_choicePref









