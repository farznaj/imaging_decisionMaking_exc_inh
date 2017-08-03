% you need the following vars.

% loadPostNameVars = 1; % set to 1 if not calling this script from
% imaging_prep_analysis


%% Set the names of imaging-related .mat file names.
% remember the last saved pnev mat file will be the pnevFileName

if loadPostNameVars
    signalCh = 2; % because you get A from channel 2, I think this should be always 2.
    % pnev2load = [];
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    [pd, pnev_n] = fileparts(pnevFileName);
    disp(pnev_n)
    cd(fileparts(imfilename))

    moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));

    [pd, pnev_n] = fileparts(pnevFileName);
    postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
end


%% Load matlab variables: event-aligned traces, inhibitRois, outcomes,  choice, etc
%     - traces are set in set_aligned_traces.m matlab script.
% Load time of some trial events

% if loadPostNameVars
%     load(postName, 'timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'time1stSideTry')

if loadPostNameVars    
    if trialHistAnalysis==0
        load(postName, 'stimAl_noEarlyDec') % Load stim-aligned_allTrials traces, frames, frame of event of interest
        stimAl_allTrs = stimAl_noEarlyDec;
    else
        load(postName, 'stimAl_allTrs')
    end
else
    stimAl_allTrs = stimAl_noEarlyDec;
end
eventI = stimAl_allTrs.eventI;
traces_al_stimAll = stimAl_allTrs.traces;
time_aligned_stim = stimAl_allTrs.time;    % '(frames x units x trials)'
fprintf('size of stimulus-aligned traces: %d %d %d (frames x units x trials)\n', size(traces_al_stimAll))

traces_al_stim = traces_al_stimAll;


%% Load outcomes and choice (allResp_HR_LR) for the current trial

if loadPostNameVars
    load(postName, 'outcomes', 'allResp_HR_LR')
end
%{
outcomes = outcomes(1:length(all_data_sess{1}));
allResp_HR_LR = allResp_HR_LR(1:length(all_data_sess{1}));
%}
choiceVecAll = allResp_HR_LR;  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.
cprintf('blue', 'Current outcome: %d correct choices; %d incorrect choices\n', sum(outcomes==1), sum(outcomes==0))
cprintf('blue','\tCorr: %d LR; %d HR\n',  sum(choiceVecAll(outcomes==1)==0), sum(choiceVecAll(outcomes==1)==1))
cprintf('blue','\tIncorr: %d LR; %d HR\n',  sum(choiceVecAll(outcomes==0)==0), sum(choiceVecAll(outcomes==0)==1))

if trialHistAnalysis
    % Load trialHistory structure to get choice vector of the previous trial
    if loadPostNameVars
        load(postName, 'trialHistory')
    end
    choiceVec0All = trialHistory.choiceVec0;
end


%% Set trials strength and identify trials with stim strength of interest

if trialHistAnalysis==0
    if loadPostNameVars
        load(postName, 'stimrate', 'cb')
    end
    
    s = stimrate-cb; % how far is the stimulus rate from the category boundary?
    if strcmp(strength2ana, 'easy')
        str2ana = (abs(s) >= (max(abs(s)) - thStimStrength));
    elseif strcmp(strength2ana, 'hard')
        str2ana = (abs(s) <= thStimStrength);
    elseif strcmp(strength2ana, 'medium')
        str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength)));
    else
        str2ana = ones(1, length(outcomes));
        
        fprintf('Number of trials with stim strength of interest = %i\n', sum(str2ana))
        %     fprintf('Stim rates for training = {}'.format(unique(stimrate[str2ana]))
    end
end


%% Set choiceVec0  (Y: the response vector)

if trialHistAnalysis
    choiceVec0 = choiceVec0All(:,iTiFlg); % choice on the previous trial for short (or long or all) ITIs
    choiceVec0S = choiceVec0All(:,1);
    choiceVec0L = choiceVec0All(:,2);
else % set choice for the current trial
    choiceVec0 = allResp_HR_LR';  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    % choiceVec0 = transpose(allResp_HR_LR);  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    if strcmp(outcome2ana, 'corr')
        choiceVec0(outcomes~=1) = nan; % analyze only correct trials.
    elseif strcmp(outcome2ana, 'incorr')
        choiceVec0(outcomes~=0) = nan; % analyze only incorrect trials.   
    end
    choiceVec0(~str2ana) = nan;   
    % Y = choiceVec0
    % print(choiceVec0.shape)
end


%% Set trsExcluded and exclude them to set X and Y; trsExcluded are trials that are nan either in traces or in choice vector.

% Identify nan trials
% trsExcluded = (sum(isnan(spikeAveEp0), 2) + isnan(choiceVec0)) ~= 0; % NaN trials % trsExcluded
trsExcluded = (sum(isnan(squeeze(nanmean(traces_al_stim, 1))'), 2) + isnan(choiceVec0)) ~= 0; % NaN trials % trsExcluded
% fprintf(sum(trsExcluded), 'NaN trials'


%% Exclude nan trials

% X = spikeAveEp0(~trsExcluded,:); % trials x neurons
Y = choiceVec0(~trsExcluded);
fprintf('%d high-rate choices, and %d low-rate choices\n', sum(Y==1), sum(Y==0))

%%
NsExcluded = false(1, size(traces_al_stim,2)); % use this if you don't want to exclude any neurons.
fprintf('chose not to remove any neurons!\n')


%%    
numTrials = sum(~trsExcluded);
numNeurons = sum(~NsExcluded);
% numNeurons = size(traces_al_stim,2);
cprintf('blue', '%d neurons; %d trials\n', numNeurons, numTrials)




