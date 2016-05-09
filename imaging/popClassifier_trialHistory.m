% This script will be called inside popClassifier if trialHistAnalysis is
% set to 1.


%% Set initial variables
%{
trialHistAnalysis = 1;
prevSuccessFlg = true; % true previous sucess trials; false: previous failure.
iTiFlg = 0; % 0: short iTi, 1: long iTi, 2: all iTis.

neuronType = 2; % 0: excitatory, 1: inhibitory, 2:all types.
alignedEvent = 'initTone'; % what event align traces on. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'

windowAvgFlg = true;
pcaFlg = false; %true;
clear epStart epEnd
% stMs = round(-300/frameLength); % the start point of the epoch relative to alignedEvent for training SVM. (500ms)
% enMs = floor(-100/frameLength); % the end point of the epoch relative to alignedEvent for training SVM. (700ms)
epStart = 1;
% epEnd = eventI;

numCVshuffIters = 10; % 100 % number of iterations for getting CV models and shuffled data.

% thAct = 1e-3; % could be a good th for excluding neurons w too little activity.
rng(0, 'twister'); % Set random number generation for reproducibility
%}


%%
doplots = true;
binRates = true;
doiti = true;
binITIs = true; % false; %true;
% vec_iti = [0 9 30]; [0 10 30]; %[0 6 9 12 30]; % [0 7 30]; % [0 10 30]; % [0 6 9 12 30]; % use [0 40]; if you want to have a single iti bin and in conventioinal analysis look at the effect of current rate on outcome.
vec_ratesdiff = 0:2:12;
% defaultHelpedTrs = false; % false; % set to 0 if you want to manually set the helped trials.
% saveHelpedTrs = true;
allowCorrectResp = 'change';
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';
% excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
% excludeExtraStim = false;
th = 5; % 10; 9; % this number of beginning trials will be excluded. also later in the code we want >th trials in each column of ratesDiffInput

if doplots
    figh = figure;
end


%% set outcome and response side for each trial, taking into account allcorrection and uncommitted responses.

% Remember if you need the outcome of previous trial, you need to use the
% final outcome of a trial (not the one before allowCorrection). This is
% why you set allowCorrectOutcomeChange to 0.
allowCorrectOutcomeChange = 0; % If 0, outcome of allowCorrEntered
% trials wont be changed (although animal's choice will be changed). If 1,
% outcome will be changed as well.
[outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp, allowCorrectOutcomeChange);
% Remember alldata.outcome is not necessarily the same as outcomes,
% depending on what you do to allCorrection trials, if it is set to change,
% then outcomes and alldata.outcome will be opposite to each other!!!


%%% set trs2rmv in the following vectors to NaN
outcomes(trs2rmv) = NaN;
% allResp(trs2rmv) = NaN;
allResp_HR_LR(trs2rmv) = NaN;


%%% set the response (current trial's choice) vector: 0 for LR and 1 for HR.
% y(tr)=1 if tr was a HR choice, y(tr)=0 if tr was a LR choice, y(tr)=nan, if tr was a trs2rmv or had an outcome other than success, failure.
y = allResp_HR_LR';

% good_corr_incorr = ~isnan(y'); % trials that are not among trs2rmv and they ended up being either correct or incorrect. (doesn't include trials with any other outcome.)
% num_final_trs = sum(good_corr_incorr), fract_final_trs = mean(good_corr_incorr) % number and fraction of trials that will be used in the model fit.



%% set variables required for logistic regression.

[~, successPrevInput, failurePrevInput, ~, ...
    itiSuccPrecedInput, itiFailPrecedInput]...
    = trialHist_logisRegress_setVar ...
    (y, outcomes, stimrate, cb, alldata, doiti, binRates, binITIs, uncommittedResp,...
    doplots, vec_ratesdiff, vec_iti, begTrs, allowCorrectOutcomeChange, th, figh);


%% Set the response vector for previous trial (allResp_HR_LR_prevTr)

switch iTiFlg
    case 0 % short ITI
        if prevSuccessFlg
            allResp_HR_LR_prevTr = itiSuccPrecedInput(:,1); % it has 1 and -1 for success HR and LR, respectively. 0 for failure. nan for any outcome other than success or failure.
            allResp_HR_LR_prevTr(allTrs2rmv) = NaN; % remove allTrs2rmv from the data
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr<0) = 0; % change the code for LR from -1 to 0
        else
            allResp_HR_LR_prevTr = itiFailPrecedInput(:,1);  % it has 1 and -1 for failure HR and LR, respectively. 0 for success. nan for any outcome other than success or failure.
            allResp_HR_LR_prevTr(allTrs2rmv) = NaN; % remove allTrs2rmv from the data
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr==0) = NaN; % set the success previous trials to Nan
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr<0) = 0; % change the code for LR from -1 to 0
        end
        
    case 1 % long ITI
        if prevSuccessFlg
            allResp_HR_LR_prevTr = itiSuccPrecedInput(:,2);
            allResp_HR_LR_prevTr(allTrs2rmv) = NaN; % remove allTrs2rmv from the data
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr<0) = 0; % change the code for LR from -1 to 0
        else
            allResp_HR_LR_prevTr = itiFailPrecedInput(:,2);
            allResp_HR_LR_prevTr(allTrs2rmv) = NaN; % remove allTrs2rmv from the data
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr==0) = NaN; % set the success previous trials to Nan
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr<0) = 0; % change the code for LR from -1 to 0
        end
        
    case 2 % all ITIs
        if prevSuccessFlg
            allResp_HR_LR_prevTr = successPrevInput;
            allResp_HR_LR_prevTr(allTrs2rmv) = NaN; % remove allTrs2rmv from the data
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr==0) = NaN; % set the failure previous trials to Nan
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr<0) = 0; % change the code for LR from -1 to 0
        else
            allResp_HR_LR_prevTr = failurePrevInput;
            allResp_HR_LR_prevTr(allTrs2rmv) = NaN; % remove allTrs2rmv from the data
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr==0) = NaN; % set the success previous trials to Nan
            allResp_HR_LR_prevTr(allResp_HR_LR_prevTr<0) = 0; % change the code for LR from -1 to 0
        end
end


%% Set Y: the response vector

choiceVec0 = allResp_HR_LR_prevTr(:);  % trials x 1;  1 for HR choice, 0 for LR prev choice.
% fprintf('N trials for LR and HR = %d  %d\n', [sum(choiceVec0==0), sum(choiceVec0==1)])


