function trs2rmv = setTrs2rmv(alldata, th, excludeExtraStim, excludeShortWaitDur, begTrs, badAlignTrStartCode, trialStartMissing)
%
% set trs2rmv (bad trials that you want to be removed from analysis)

%{
if exist('trials_per_session', 'var') && ~isempty(trials_per_session)
    % index of trial 1 in the concatenated alldata for each session
    begTrs = [0 cumsum(trials_per_session)]+1;
    begTrs = begTrs(1:end-1);
else
    begTrs = 1;
end
%}

if ~exist('begTrs', 'var')
    begTrs = 1;
end


%%
% th = 5; % 9; % this number of beginning trials will be excluded. also later in the code we want >th trials in each column of ratesDiffInput
b = bsxfun(@plus, begTrs, (0:th-1)'); b = b(:); % don't use the 1st 10 trials.
trs_begWarmUp = b;

%{
% trs2rmv = unique([trs_problemAlign, trs_badMotion_pmtOff]);
if excludeExtraStim
    trs_begWarmUp = begTrs;
else % make sure you don't use the 1st 10 trials.
    b = bsxfun(@plus, begTrs, (0:th)'); b = b(:);
    trs_begWarmUp = b;
end
%}

trs_helpedInit = find([alldata.helpedInit]==1);
trs_helpedChoice = find([alldata.helpedChoice]==1);

trs_extraStim = find([alldata.extraStimDuration] > 0);

waitdur_th = .032; % sec
trs_shortWaitdur = find([alldata.waitDuration] < waitdur_th)';

trs2rmv = unique([trs_begWarmUp(:); trs_helpedInit(:); trs_helpedChoice(:)]); %, trs_unwantedOutcome]);
if excludeExtraStim
    disp('Excluding trials with extraStim!')
    trs2rmv = unique([trs2rmv(:); trs_extraStim(:)]);
end
if excludeShortWaitDur
    disp('Excluding trials with waitDur < 32ms!')
    trs2rmv = unique([trs2rmv(:); trs_shortWaitdur(:)]);
end

% len_fract_trs2rmv = [length(trs2rmv) length(trs2rmv)/length(alldata)]


%%
if exist('badAlignTrStartCode', 'var')
    % Ttrials that alignment of behavior and imaging cannot be performed, i.e. badAlignTrStartCode, and trialStartMissing trials
    trs_problemAlign = unique([find(badAlignTrStartCode)==1, find(trialStartMissing)==1]);
    
    % Trials including pmtOffFrames and bigMotion (badFrames).
    trs_badMotion_pmtOff = unique([find([alldata.anyBadFrames]==1), find([alldata.anyPmtOffFrames]==1)]);
    
    
    trs2rmv = unique([trs2rmv; trs_problemAlign(:); trs_badMotion_pmtOff(:)]);
end


