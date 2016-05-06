function [stimrate, y, HRchoicePerc] = stimrate_choice_set...
    (alldata, trials_per_session, uncommittedResp, allowCorrectResp, ...
    excludeExtraStim, excludeShortWaitDur, mouse)
% set stimrate and choice (allResp) for pooled trials of a mouse.
% called inside performance_trainingDays.m

%%
if exist('trials_per_session', 'var') && ~isempty(trials_per_session)
    % index of trial 1 in the concatenated alldata for each session
    begTrs = [0 cumsum(trials_per_session)]+1;
    begTrs = begTrs(1:end-1);
else
    begTrs = 1;
end

if ~exist('uncommittedResp', 'var')
    uncommittedResp = 'nothing'; % leave outcome and allResp as they are (ie go with the final choice of the mouse).
end

if ~exist('allowCorrectResp', 'var')
    allowCorrectResp = 'change'; % change the response on trials that entered allowCorrection to the original choice.
end

if ~exist('excludeExtraStim', 'var')
    excludeExtraStim = false;
end

if ~exist('excludeShortWaitDur', 'var')
    excludeShortWaitDur = false;
end


if ~exist('mouse', 'var') || isempty(mouse)
    mouse = '';
end


%% set trs2rmv (bad trials that you want to be removed from analysis)

th = 5; % 10; 9; % this number of beginning trials will be excluded. also later in the code we want >th trials in each column of ratesDiffInput
trs2rmv = setTrs2rmv(alldata, th, excludeExtraStim, excludeShortWaitDur, begTrs);
% Use the code below if you are analyzing imaging data.
% trs2rmv = setTrs2rmv(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, badAlignTrStartCode, trialStartMissing, trialCodeMissing);

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
    %     B = []; deviance = []; stats = [];
    %     X = []; y = []; vec_ratesdiff = []; ratesDiffVecInds = []; vec_iti = []; itiVecInds = [];
    %     ratesDiffInput = []; successPrevInput = []; failurePrevInput = []; itiPrecedInput = [];
    %     return
    
    % elseif doplots
    %     figh = figure('name', ['Mouse ', mouse]);
end


%%
stimrate(isnan(y)) = nan;




%%
% if doplots

plotPMF = false;
shownumtrs = false; %true;

HRchoicePerc = PMF_set_plot(stimrate, y, cb, [], plotPMF, shownumtrs);


% end

