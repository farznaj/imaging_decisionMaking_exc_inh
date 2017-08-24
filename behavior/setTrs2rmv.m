function trs2rmv = setTrs2rmv(alldata, th, excludeExtraStim, excludeShortWaitDur, begTrs, badAlignTrStartCode, trialStartMissing, trialCodeMissing, trStartMissingUnknown, trEndMissing, trEndMissingUnknown, trialNumbers)
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

if ~exist('trEndMissing', 'var'), trEndMissing = []; end
if ~exist('trEndMissingUnknown', 'var'), trEndMissingUnknown = []; end
if ~exist('trStartMissingUnknown', 'var'), trStartMissingUnknown = []; end
if ~exist('trialNumbers', 'var'), trialNumbers = []; end
% if ~exist('pmtOffTrials', 'var'), pmtOffTrials = []; end


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
else
    disp('Not excluding trials with extraStim!')
end

if excludeShortWaitDur
    fprintf('Excluding %i trials with waitDur < 32ms!\n', length(trs_shortWaitdur(:)))
    trs2rmv = unique([trs2rmv(:); trs_shortWaitdur(:)]);
else
    disp('Not excluding trials with waitDur < 32ms!')
end

% len_fract_trs2rmv = [length(trs2rmv) length(trs2rmv)/length(alldata)]


%% If you care about imaging data, then you need to run this part.
% but if it is all behavior, you don't need to run this section.

if exist('badAlignTrStartCode', 'var')
    % Ttrials that alignment of behavior and imaging cannot be performed, i.e. badAlignTrStartCode, trialStartMissing and trStartMissingUnknown trials
    trs_problemAlign = unique([find(badAlignTrStartCode==1), find(trialStartMissing==1), trialNumbers(trStartMissingUnknown)]);
    
    % Trials including pmtOffFrames and bigMotion (badFrames).
    trs_badMotion_pmtOff = unique([find([alldata.anyBadFrames]==1), find([alldata.anyPmtOffFrames]==1)]);
    
    % Trials that miss frames at the end (they are fine for alignments on earlier events)
    trs_endMiss = unique([trialNumbers(trEndMissing), trialNumbers(trEndMissingUnknown)]);
    
    % Trials that miss trialCode signal.
    trs_noTrialCode = find(trialCodeMissing==1);    
%     trs_noTrialCode(trs_noTrialCode>length(alldata)) = []; 
    
    % Trials that were not imaged. (I believe trs_noTrialCode is a subset of trs_notScanned).
    trs_notScanned = find([alldata.hasActivity]==0);    
    if ~isequal(trs_noTrialCode, trs_notScanned), warning('I am curious why and how these two variables are different!'), end
    
    
    trs2rmv = unique([trs2rmv; trs_problemAlign(:); trs_badMotion_pmtOff(:); trs_noTrialCode(:); trs_notScanned(:); trs_endMiss(:)]);
    if sum(isnan(trs2rmv)) > 0
        warning('trs2rmv includes nan; must be bc of a nan in trialNumbers, due to scanned trial but missed trialCode. TrialCodeMissing includes this trial, so removing nan from trs2rmv.')
        trs2rmv(isnan(trs2rmv)) = [];
    end

end


