% This is the main and starting script for the analysis of your imaging data. It gives
% you the vars that you need for further analyses.

% pnev_manual_comp_setVars is a very nice script (together with
% pnev_manual_comp_match) that allows you to plot and compare the trace and
% ROIs of 2 different methods.
%
% you can use it to compare Eftychios vs manual. Or 2 different channels.
% Or 2 different methods of Eftychios, etc.


%% worry about bleedthrough, and visual artifact.

%%
% outName = 'fni17-151016';
mouse = 'fni17';
imagingFolder = '151029'; % '151021';
mdfFileNumber = 3; % or tif major
signalCh = 2;

pnev2load = []; %7 % 4

setInhibitExcit = true; % if 1, inhibit and excit traces will be set.
    sigTh = 1.2; % signal to noise threshold for identifying inhibitory neurons on tdtomato channel. eg. sigTh = 1.2;
    showResults = false; % set to true so u can evaluate identification of inhibitory neurons (ie if sigTh is doing a good job).
    
compareManual = false; % compare results with manual ROI extraction
plotTraces1by1 = false; % plot traces per neuron and per trial showing all trial events

autoTraceQual = 1; % if 1, automatic measure for trace quality will be used.
examineTraceQual = 0; % if 0, traceQuality array needs to be saved.
    saveTraceQual = 0; % it will only take effect if examineTraceQual is 1.
    analyzeQuality = [1 2]; % 1(good) 2(ok-good) 3(ok-bad) 4(bad) % trace qualities that will be analyzed.
orderTraces = 0; % if 1, traces will be ordered based on the measure of quality from high to low quality.

analyzeOutcomes = {'all'}; % {'success', 'failure'}; % outcomes that will be analyzed.

excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;
allowCorrectResp = 'change'; % 'change'; 'remove'; 'nothing'; % if 'change': on trials that mouse corrected his choice, go with the original response.
uncommittedResp = 'nothing'; % 'change'; 'remove'; 'nothing'; % what to do on trials that mouse made a response (licked the side port) but did not lick again to commit it.

thbeg = 5; % n initial trials to exclude.

helpedInit = []; % [100];
helpedChoice = []; %31;
defaultHelpedTrs = 0; % if 1, the program assumes that no trial was helped.
saveHelpedTrs = 0; % it will only take effect if defaultHelpedTrs is false. If 1, helpedTr fields will be added to alldata.

setNaN_goToneEarlierThanStimOffset = 0; % if 1, set to nan eventTimes of trials that had go tone earlier than stim offset... if 0, only goTone time will be set to nan.


%%
frameLength = 1000/30.9; % sec.

% remember if there are 
% set the names of imaging mat files (one includes pnev results and one
% includes all other vars)
% remember the last saved pnev mat file will be the pnevFileName

[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[~,f] = fileparts(pnevFileName);
disp(f)
cd(fileparts(imfilename))


% load alldata 
% load(imfilename, 'all_data'), all_data = all_data(1:end-1);   %from the imaging file.  % alldata = removeBegEndTrs(alldata, thbeg);
% use the following if you want to go with the alldata of the behavior folder
% set filenames
[alldata_fileNames, ~] = setBehavFileNames(mouse, {datestr(datenum(imagingFolder, 'yymmdd'))});
% sort it
[~,fn] = fileparts(alldata_fileNames{1});
a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
[~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
alldata_fileNames = alldata_fileNames(isf);
% load the one corresponding to mdffilenumber.
[all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber)); % , defaultHelpedTrs, saveHelpedTrs); % it removes the last trial too.
fprintf('Total number of behavioral trials: %d\n', length(all_data))


% begTrs = [0 cumsum(trials_per_session)]+1;
% begTrs = begTrs(1:end-1);
begTrs = 1; % 1st trial of each session
    
% use the following plot to decide on thbeg: number of begining trials that
% you want to exclude.
% stimDur_diff
figure; hold on
title(sprintf('thbeg = %d', thbeg))
plot([all_data.waitDuration]+[all_data.postStimDelay])
plot([all_data.stimDuration])
plot([all_data.extraStimDuration])
plot([all_data.stimDur_aftRew])
plot([all_data.stimDur_diff])
plot([thbeg thbeg],[-1 12], 'k')
legend('waitDur', 'stimDuration', 'extraStimDur', 'stimDur\_aftRew', 'stimDur\_diff')
set(gcf, 'name', 'stimDuration= stimDur_diff (or if it is 0, waitDur) + extrastim_dur + stimdur_aftrew .')
m = max([all_data.stimDuration]);
ylim([-1 m+1])
set(gca,'tickdir','out')

%{
- stimulus is played for stimDuration which equal stimDur_diff (or if it is 0, waitDur) + extrastim_dur + stimdur_aftrew .
- total waitdur (ie since stim onset, when mouse was allowed to do cent commit) equals waitDuration + postStimDelay
%}


load(imfilename, 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'badFrames', 'pmtOffFrames', 'trialCodeMissing')
trialNumbers(isnan(framesPerTrial)) = [];
frame1RelToStartOff(isnan(framesPerTrial)) = [];
framesPerTrial(isnan(framesPerTrial)) = [];

if ~exist('trialCodeMissing', 'var') % for those days that you didn't save this var.
    trialCodeMissing = [];
end

fprintf('Total number of imaged trials: %d\n', length(trialNumbers))
if ~any(trialCodeMissing)
    fprintf('All trials are triggered :)\n')
else
    fprintf('There are non-triggered trials! Trial %d\n', find(trialCodeMissing))
end





%%
%%%%%%%%%%%%%%%%%%%%%%%%% Load neural data, merge them into all_data, and assess trace quality %%%%%%%%%%%%%%%%%%%%%%%

%% Load vars related to manual method

if compareManual
    
    load(imfilename, 'imHeight', 'imWidth', 'pmtOffFrames')
    load(pnevFileName, 'activity_man_eftMask')
%     load(imfilename, 'activity_man_eftMask')
    
    %{
    load(imfilename, 'rois', 'activity') % , 'imHeight', 'imWidth', 'pmtOffFrames')
    load(pnevFileName, 'A') % load('demo_results_fni17-151102_001.mat', 'A2')
    spatialComp = A; % A2 % obj.A; 
    clear A
    %}
    
    %% manual activity and dfof
    activity_man = activity_man_eftMask;

%     activity_man = activity;
    clear activity activity_man_eftMask

    % Compute df/f for the manually found activity trace.
    gcampCh = 2;
    smoothPts = 6;
    minPts = 7000; %800;
    dFOF_man = konnerthDeltaFOverF(activity_man, pmtOffFrames{gcampCh}, smoothPts, minPts);


    %% masks and matching ROIs between the 2 methods.
%     csfrs = [0 cumsum(framesPerTr)];
    
    % if you computed manual activity on eft mask:
    matchedROI_idx = 1:size(dFOF_man,2); % there is a 1-1 match between ROIs
    
    % set contours for Eft ROIs
    %{
    contour_threshold = .95;
    [CCorig, CR, COMs] = ROIContoursPnev(spatialComp, imHeight, imWidth, contour_threshold); % P.d1, P.d2
    CC = ROIContoursPnev_cleanCC(CCorig);    
    % set masks for Eft ROIs
    mask_eft = maskSet(CC, imHeight, imWidth);
    
    
    % set masks for manual ROIs
    CC_rois = cell(1, length(rois)); % Set the coordinates for the manual method in [y x] format for each ROI. (similar to CC of Eftychios method.)
    for rr = 1:length(rois)
        CC_rois{rr} = [rois{rr}.mnCoordinates(:, 2)'; rois{rr}.mnCoordinates(:, 1)'];
    end    
    mask_manual = maskSet(CC_rois, imHeight, imWidth);
    
    % find manual ROIs that match Eft ROIs
    refMask = mask_eft;
    toMatchMask = mask_manual;
    matchedROI_idx = matchROIs_sumMask(refMask, toMatchMask); % matchedROI_idx(i)=j means ROI i of refMask matched ROI j of toMatchMask.
    %}
    
end


%% Set spikes, activity, dFOF by loading vars from Eftychios output, and merge them into alldata

load(pnevFileName, 'C', 'C_df', 'options') % , 'S')
if strcmp(options.deconv_method, 'MCMC')
    load(pnevFileName, 'S') % if mcmc
    spiking = S; % S; % S_mcmc; % S2;
elseif strcmp(options.deconv_method, 'constrained_foopsi')    
    load(pnevFileName, 'S_df') % if constrained foopsi
    spiking = S_df; % S; % S_mcmc; % S2;
end
spikes = spiking'; % frames x units

if size(C,2) ~= size(C_df,2) % iti-nans were inserted in C and S: remove them.
    load(imfilename, 'Nnan_nanBeg_nanEnd')
    nanBeg =  Nnan_nanBeg_nanEnd(2,:);
    nanEnd = Nnan_nanBeg_nanEnd(3,:);
    inds2rmv = cell2mat(arrayfun(@(x,y)(x:y), nanBeg, nanEnd, 'uniformoutput', 0)); % index of nan-ITIs (inferred ITIs) on C and S traces.
    C(:, inds2rmv) = [];    
   
    if size(spikes,1) ~= size(C_df,2)
        spikes(inds2rmv,:) = [];    
    end
end

an = find(~sum(~isnan(C),2));
if ~isempty(an)
    warning(sprintf(['C trace of neuron(s) ', repmat('%i ', 1, length(an)), 'is all NaN. This should not happen!'], an));
end
an = find(~sum(~isnan(C_df),2));
if ~isempty(an)
    warning(sprintf(['C_df trace of neuron(s) ', repmat('%i ', 1, length(an)), 'is all NaN. This should not happen!'], an));
end

% load('demo_results_fni17-151102_001.mat', 'C_mcmc', 'C_mcmc_df', 'S_mcmc')

% spatialComp = A2; % obj.A; 
temporalComp = C; % C_mcmc; % C2; % obj.C; 
temporalDf = C_df; % C_mcmc_df; % C_df; % obj.C_df; 

% spikingDf = S_df; % obj.S_df;
clear C C_df S % ('C_mcmc', 'C_mcmc_df', 'S_mcmc')

activity = temporalComp'; % frames x units
% activity = activity_man;
% Sdf = S_df';

if size(temporalDf,1) == size(temporalComp,1)+1
    temporalDf(end,:) = [];
end
dFOF = temporalDf';
% dFOF = dFOF_man;
clear temporalComp temporalDf spiking


%% Merge imaging variables into all_data (before removing any trials): activity, dFOF, spikes

minPts = 7000; %800;
[all_data, mscanLag] = mergeActivityIntoAlldata_fn(all_data, activity, framesPerTrial, ...
  trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, dFOF, spikes);

% manual
% activity = activity_man;
% dFOF = dFOF_man;
% [all_data, mscanLag] = mergeActivityIntoAlldata_fn(all_data, activity_man, framesPerTrial, ...
%   trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, dFOF_man, spikes);

%{
%% Take care of helped trials: you don't need it. you've done it for all ur behavioral data and appended the helped fields to alldata and saved it. if not the following function will do it.
% [alldata_fileNames, ~] = setBehavFileNames(mouse, {datestr(datenum(imagingFolder, 'yymmdd'))});
% [all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber), defaultHelpedTrs, saveHelpedTrs);

% If defaultHelpedTrs is false and alldata doesn't include the helped fields, add helpedInit and helpedChoice fields to alldata, and save it.

alldata_fileNam = imfilename;
all_data = setHelpedTrs(all_data, defaultHelpedTrs, saveHelpedTrs, alldata_fileNam, helpedInit, helpedChoice);
%}
  

%% Take care of trials that are in alldata but not in imaging data.

%{
if sum(~ismember(1:length(trialNumbers), trialNumbers))~=0
    error('There are non-triggered trials in mscan! Decide how to proceed!')
    % try this:
    % remove the trial that was not recorded in MScan
    all_data(~ismember(1:length(trialNumbers), trialNumbers)) = []; % FN note: double check this.
end
alldata = all_data(trialNumbers); % in case mscan crashed, you want to remove trials that were recorded in bcontrol but not in mscan.
%}

alldata = all_data;

% clear all_data
% alldata = alldata(1:end-1);  % you commented it here and added it above. loadBehavData by default removes the last trial.   % alldata = removeBegEndTrs(alldata, thbeg);
% fprintf('Total number of imaged trials: %d\n', length(alldata))


% Add NaN for alldata.dFOF, spikes, and activity of those trials that were
% not imaged (either at the end, or at the middle due to mscan failure.)

% For trials that were not imaged, set frameTimes, dFOF, spikes and activity traces to all nans (size:
% min(framesPerTrial) x #neurons).
[alldata([alldata.hasActivity]==0).dFOF] = deal(NaN(min(framesPerTrial), size(dFOF,2)));
[alldata([alldata.hasActivity]==0).spikes] = deal(NaN(min(framesPerTrial), size(dFOF,2)));
[alldata([alldata.hasActivity]==0).activity] = deal(NaN(min(framesPerTrial), size(dFOF,2)));
[alldata([alldata.hasActivity]==0).frameTimes] = deal(NaN(1, min(framesPerTrial)));


%% Assess trace quality of each neuron

if examineTraceQual    
%     assessCaTraceQaulity % another script to look at traces 1 by 1.

    inds2plot = randperm(size(activity,2));
    traceQuality = plotEftManTracesROIs(dFOF', spikes', [], [], [], [], [], [], [], activity', inds2plot, 1, 0, []);

    if saveTraceQual
        save(imfilename, 'traceQuality', '-append')
    end
    goodNeurons = ismember(traceQuality, analyzeQuality);
end

    
%% Automatic assessment of trace quality

% Compute measures of trace quality
if autoTraceQual
    [avePks2sdS, aveProm2sdS, measQual] = traceQualMeasure(dFOF', spikes');

    % Sort traces
    [sPks, iPks] = sort(avePks2sdS);
    [sProm, iProm] = sort(aveProm2sdS);
    [sPksProm, iPksProm] = sort(avePks2sdS .* aveProm2sdS);
    [sMeasQual, iMeasQual] = sort(measQual);

    % badQual = find(avePks2sdS<3 | aveProm2sdS<1);
    badQual = iMeasQual(sMeasQual<0);
    % look at badQual neurons
%     plotCaTracesPerNeuron({dFOF(:, badQual)}, 0, 0, 1, [], []) 

    goodNeurons = true(1,length(avePks2sdS));
    goodNeurons(badQual) = false;
    
    % order based on automatic measure
    goodnessOrder = iMeasQual(sMeasQual>=0); % ascending order of neurons based on trace quality
    goodnessOrder = goodnessOrder(end:-1:1); % descending order
end
% fprintf('N good-quality and all neurons: %d, %d. Ratio: %.2f\n', [sum(goodNeurons), size(activity,2), sum(goodNeurons)/ size(activity,2)]); % number and fraction of good neurons vs number of all neurons.


% set goodinds: an array of length of all neurons, with 1s indicating good and 0s bad neurons.
if ~any([autoTraceQual, examineTraceQual])  % if no trace quality examination (auto or manual), then include all neurons.
    goodinds = true(1, size(dFOF,2)); %    goodnessOrder = 1:size(dFOF,2);    goodNeurons = 1:size(dFOF,2);
elseif orderTraces % descending based on the measure of quality
    error('the code below needs work. goodinds is supposed to be logical not tr numbers.')
    goodinds = goodnessOrder; 
else % stick to the original order
    goodinds = goodNeurons;
end

fprintf('N good-quality and all neurons: %d, %d. Ratio: %.2f\n', [sum(goodinds), size(activity,2), sum(goodinds)/ size(activity,2)]); % number and fraction of good neurons vs number of all neurons.






%%
%%%%%%%%%%%%%%%%%%%%%%%%% Take care of behavioral data (all_data) %%%%%%%%%%%%%%%%%%%%%%%

%% Set problematic trials to later exclude them if desired.

% commenting for now but you may need it later.
%{
% in set_outcome_allResp you can take care of the following terms... u
% don't need to worry about them here.
% Trials that the mouse entered the allow correction state.
a = arrayfun(@(x)x.parsedEvents.states.punish_allowcorrection, alldata, 'uniformoutput', 0);
trs_allowCorrectEntered = find(~cellfun(@isempty, a)); 

% Trials that mouse licked the error side during the decision time. Afterwards, the mouse may have committed it
% (hence entered allow correction) or may have licked the correct side.
a = arrayfun(@(x)x.parsedEvents.states.errorlick_again_wait, alldata, 'uniformoutput', 0);
trs_errorlick_again_wait_entered = find(~cellfun(@isempty, a));


% Trials with unwanted outcomes.
% suc*, fail*, early*, no dec*/cho*, wrong st*/in*, no cen* com*, no s* com
% labels = [1 0 -1 -2 -3 -4 -5];
if strcmp(analyzeOutcomes, 'all')
    trs_unwantedOutcome = [];
else
    [~, trs_unwantedOutcome] = trialsToAnalyze(alldata, analyzeOutcomes);
end


% Trials during which mouse was running.
trs_mouseRunning = [];

%{
thWheel = .05;
wheelRevolution = {alldata.wheelRev};
rangeWheelRev = cellfun(@range, wheelRevolution);

max(rangeWheelRev)
min(rangeWheelRev)
mean(rangeWheelRev)

trs_mouseRunning = find(rangeWheelRev > thWheel); % trials in which mouse moved more than thWheel
length(trs_mouseRunning)
% trs2rmv = [];
%}

%}

%% Set trs2rmv, stimrate, outcome and response side. You will set certain variables to NaN for trs2rmv (but you will never remove them from any arrays).

% warning('You have to specify trs2rmv!!')

cb = unique([alldata.categoryBoundaryHz]); % category boundary in hz

load(imfilename, 'badAlignTrStartCode', 'trialStartMissing'); %, 'trialCodeMissing') % they get set in framesPerTrialStopStart3An_fn

% set the following as trs2rmv: trs_begWarmUp(:); trs_helpedInit(:); trs_helpedChoice(:), trs_extraStim, trs_shortWaitdur, trs_problemAlign, trs_badMotion_pmtOff 
trs2rmv = setTrs2rmv(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, badAlignTrStartCode, trialStartMissing, trialCodeMissing);


%%%%%%% Set stim rate: trials with stimrate=nan, must be added to trs2rmv as well
[~, stimdur, stimrate, stimtype] = setStimRateType(alldata); % stimtype = [multisens, onlyvis, onlyaud];
if sum(isnan(stimrate)) > 0
    fprintf('# trials with NaN stimrate= %i\n', sum(isnan(stimrate)))
    trs2rmv = unique([trs2rmv; find(isnan(stimrate))]);
end

%{
trs2rmv = unique([trs2rmv, trs_unwantedOutcome]);

% strictly correct trials: no allow correction, no error lick (even if not committed).
trs2rmv = unique([trs2rmv, trs_allowCorrectEntered, trs_errorlick_again_wait_entered]);
%}


%%%%%%% take care of cases that go tone happened earlier than stim offset
%%%%%%% and resulted in a different stim type.
fractFlashBefGoTone = NaN(1, length(alldata));
stimRateChanged = false(1, length(alldata));
goToneRelStimOnset = NaN(1, length(alldata));
for tr = 1:length(alldata)
    if ~isempty(alldata(tr).parsedEvents.states.center_reward) % ~isnan(goToneRelStimOnset(tr))
        % xx shows flashes/clicks onset time during the stimulus
        xx = cumsum(alldata(tr).auditoryIeis + alldata(tr).eventDuration);
        xx = [0 xx(1:end-1)] * 1000 + 1; % ms % 1st flash onset is assigned 1


        % goToneRelStimOnset = timeCommitCL_CR_Gotone - timeStimOnset + 1;
        goToneRelStimOnset(tr) = ((alldata(tr).parsedEvents.states.center_reward(1))*1000 - (alldata(tr).parsedEvents.states.wait_stim(1))*1000) + 1;

        
        % percentage of flashes/clicks that happened before go tone. If 1,
        % the animal received the whole stimulus before go tone.
        fractFlashBefGoTone(tr) = sum(xx < goToneRelStimOnset(tr)) / length(xx); 
        
        % did stim type (hr vs lr) at go tone change compared to the
        % original stimulus?
        if sign(length(xx)-cb) ~= sign((sum(xx < goToneRelStimOnset(tr)))-cb) % alldata(tr).nAuditoryEvents
            stimRateChanged(tr) = true;
        end
    end
end

% find(fractFlashBefGoTone<1) % in these trials go tone came before the entire flashes were played.
% fractFlashBefGoTone_ifLessThan1 = fractFlashBefGoTone(fractFlashBefGoTone<1);
% fprintf('Fract flash before goTone if <1\n%.2f \n', fractFlashBefGoTone(fractFlashBefGoTone<1))
if sum(fractFlashBefGoTone<1)>0
%     nanmean(fractFlashBefGoTone(fractFlashBefGoTone<1))
    aveFractFlashBefGoTone_forLessThan1 = nanmean(fractFlashBefGoTone(fractFlashBefGoTone<1))
    
    figure; plot(fractFlashBefGoTone)
    xlabel('Trial')
    ylabel([{'Fraction of flashes played before Go Tone'}, {'Ideally we want 1.'}])
end


if any(stimRateChanged)
    trsStimTypeScrewed = find(stimRateChanged)'; % in these trials when go tone came stim type (hr vs lr) was different from the entire stim... these are obviously problematic trials.
    fprintf('%d = #trs with a different stim type at Go Tone than the actual stim type\n', length(trsStimTypeScrewed))
    aveFractFlashBefGoTone_stimRateChanged = nanmean(fractFlashBefGoTone(stimRateChanged))
    aveOutcomeFractFlashBefGoTone_stimRateChanged = nanmean([alldata(stimRateChanged).outcome])

    trs2rmv = unique([trs2rmv; trsStimTypeScrewed]);
end

fprintf('Number of trs2rmv: %d\n', length(trs2rmv))
disp(trs2rmv')

%%%%% Set outcome and response side for each trial, taking into account allcorrection and uncommitted responses.

% Set some params related to behavior % behavior_info
[outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp);

% set trs2rmv to nan
outcomes(trs2rmv) = NaN; 
allResp(trs2rmv) = NaN; 
allResp_HR_LR(trs2rmv) = NaN;
% stimrate(trs2rmv) = NaN;


% save('151102_001.mat', '-append', 'trs2rmv')  % Do this!


%% Set event times (ms) relative to when bcontrol starts sending the scope TTL. event times will be set to NaN for trs2rmv.

[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv);

trsGoToneEarlierThanStimOffset = find(timeCommitCL_CR_Gotone < timeStimOffset)';


%% Take care of trials that mouse received center reward and go tone before the
% end of the stimulus. This is important if looking at neural responses
% during stim interval: you dont want to look beyond goTone time for these
% trials... not necessarily important for other alignments. So you may or
% may not need to run this part! If want to be conservative, run it!

% but also remember when you align traces on 1stSideTry you don't want to
% have stimulus played during the baseline. Usually 1stSideTry happens
% long after stimOffset so this shouldn't be a problem. Figures below help
% you dig into this. If you want to be very careful, you should remove
% trials that have stim during n frames before time1stSideTry where n =
% nPreFrame for alignments on 1stSideTry. Below you are removing trials
% with stimOffset during the 3 frames before 1stSideTry.

if isempty(trsGoToneEarlierThanStimOffset)
    fprintf('%d = No trials withgoTone earlier than stimOffset :)\n')
else

    figure('name', 'Trials with goTone earlier than stimulus offset'); 
    subplot(211), hold on
    plot(timeStimOffset(trsGoToneEarlierThanStimOffset)  -  ...
        timeCommitCL_CR_Gotone(trsGoToneEarlierThanStimOffset))
    
    plot(time1stSideTry(trsGoToneEarlierThanStimOffset)  -  ...
        timeStimOffset(trsGoToneEarlierThanStimOffset))    
    plot([1 length(trsGoToneEarlierThanStimOffset)], [3*frameLength 3*frameLength], 'k:')
    ylabel('Time (ms)')
    legend('time after goTone with stimulus', 'time before 1stSideTry without stimulus') % this is a more understandable legend
%     legend('stimOffset - goTone', '1stSideTry - stimOffset')
    
    subplot(212), hold on; 
    plot(timeCommitCL_CR_Gotone(trsGoToneEarlierThanStimOffset))
    plot(timeStimOffset(trsGoToneEarlierThanStimOffset), 'r')
    plot(time1stSideTry(trsGoToneEarlierThanStimOffset), 'g')
    plot(timeReward(trsGoToneEarlierThanStimOffset), 'm')
    plot(timeCommitIncorrResp(trsGoToneEarlierThanStimOffset), 'c')
    xlabel('Trials with Go tone earlier than stim offset')
    ylabel('Time (ms)')
    legend('goTone', 'stimOffset', '1stSideTry', 'Reward', 'commitIncorr')
    
    
    %%
    fprintf('%d = #trs goTone earlier than stimOffset\n', length(trsGoToneEarlierThanStimOffset))
    fprintf('Removing them from timeCommitCL_CR_Gotone!\n')    
    % timeStimOnset(trsGoToneEarlierThanStimOffset) = NaN; % IMPORTANT: you dont want to look beyond goTone time for these trials, but upto go tone is safe!
    timeCommitCL_CR_Gotone(trsGoToneEarlierThanStimOffset) = NaN;  % you want this for sure (bc in some trials stim has been still playing after go tone).

    
    %%
    a = time1stSideTry(trsGoToneEarlierThanStimOffset) - timeStimOffset(trsGoToneEarlierThanStimOffset);
    if sum(a < 3*frameLength)
        fprintf('There are %i trials with stimOffset within 3 frames before 1stSideTry.\n', sum(a < 3*frameLength))
        fprintf('Removing them from 1stSideTry, 1stCorrTry, 1stIncorrTry!\n')
        aa = trsGoToneEarlierThanStimOffset(a < 3*frameLength);
        time1stSideTry(aa) = NaN;
        time1stCorrectTry(aa) = NaN;
        time1stIncorrectTry(aa) = NaN;
        time1stCorrectResponse(aa) = NaN;
    end
    
    
    %%
    % For the following events not as important to set to nan unless you think
    % neural responses aligned on these events can be different when go tone
    % happened during stim, which is likely!
    if setNaN_goToneEarlierThanStimOffset % if 1, set to nan eventTimes of trials that had go tone earlier than stim offset... if 0, only goTone time will be set to nan.
        timeStimOnset(trsGoToneEarlierThanStimOffset) = NaN; % you dont want to look beyond goTone time for these trials, but upto go tone is safe!
        timeStimOffset(trsGoToneEarlierThanStimOffset) = NaN; 
        time1stSideTry(trsGoToneEarlierThanStimOffset) = NaN;  
        time1stCorrectTry(trsGoToneEarlierThanStimOffset) = NaN; 
        time1stIncorrectTry(trsGoToneEarlierThanStimOffset) = NaN; 
        timeReward(trsGoToneEarlierThanStimOffset) = NaN; 
        timeCommitIncorrResp(trsGoToneEarlierThanStimOffset) = NaN; 
        time1stCorrectResponse(trsGoToneEarlierThanStimOffset) = NaN; 
    end
    
    
end


% if you want to be super conservative, just reset trs2rmv and again set the times:
% trs2rmv = unique([trs2rmv; trsGoToneEarlierThanStimOffset]);
% [timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
%     time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
%     setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv);




%%% Done: Important: take care of the issue that mouse may have made his decision before the end of the stimulus.
% make sure when looking at the stimulus epoch, it is all
% stimulus and does not include choice or episodes of stimulus absence!
%
% timeStimOffset in setEventTimesRelBcontrolScopeTTL.m accurately shows
% stim offset. 
% but mouse may have committed center lick (hence received cent reward and
% go tone) before stim offset.
% timeCommitCL_CR_Gotone shows the time of commit cent lick and go tone, so
% if timeCommitCL_CR_Gotone < timeStimOffset, go tone (and cent reward)
% happened before the stim was over, so you need to exclude these trials.
%
% this is bc not only stim interval includes cent reward and go tone, but
% also stim rate might be different at the time of go tone than what the
% final rate which determins the reward.
%
% [(1:length(timeCommitCL_CR_Gotone))' timeCommitCL_CR_Gotone' , timeStimOffset' , [timeCommitCL_CR_Gotone < timeStimOffset]']

%{
% this plot shows when flashes/clicks happened (their onset, remember they
% last for alldata.eventDuration) during the stimulus
yy = ones(1, length(alldata(tr).auditoryIeis));
figure; plot(xx, yy, 'o')

% Not useful, what matters is when the go tone was played not when waitdur ended: how many events were played (ie their onset had happened) when the waitDur happened.
% sum(xx < alldata(tr).waitDuration * 1000)

diff([[alldata.waitDuration]', [alldata.stimDuration]'], [], 2)

% trials with different stimdur_diff than waitdur
if any([alldata.stimDur_diff])
    [alldata.stimDur_diff] ~= [alldata.waitDuration]
end
%}




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Take care of neural traces in alldata %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% In alldata, set traces and choose good quality traces.

% alldataDfofGood = cellfun(@(x)x(:, goodinds), {alldata.activity}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
alldataDfofGood = cellfun(@(x)x(:, goodinds), {alldata.dFOF}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
alldataSpikesGood = cellfun(@(x)x(:, goodinds), {alldata.spikes}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.

activityGood = activity(:, goodinds); % frames x units % remember activity and dFOF may have more frames that alldataDfofGood_mat bc alldataDfofGood_mat does not include the frames of the trial during which mscan was stopped but activity includes those frames.
dfofGood = dFOF(:, goodinds); % frames x units    
spikesGood = spikes(:, goodinds); % frames x units

alldataDfofGood_mat = cell2mat(alldataDfofGood'); % frames x units;  % same as dfofGood except for some frames at the end that it may miss due to stopping mscan at the middle of a trial.

if compareManual
    eftMatchIdx_mask_good = matchedROI_idx(goodinds);
end

% alldataDfof = {alldata.dFOF};
% alldataDfof_mat = cell2mat(alldataDfof'); % frames x units


numUnits = size(alldata(1).dFOF,2); % all units, regardless of the quality.
numGoodUnits = size(dfofGood,2);
numTrials = length(alldata);


%% Set inhibit and excit traces.

% alldataDfofGoodInh, alldataSpikesGoodInh and alldataDfofGoodExc, alldataSpikesGoodExc.
% size: 1 x number of trials, each element includes the traces of all inh (exc)
% neurons. 
if setInhibitExcit

    [inhibitRois, good_inhibit, good_excit] = inhibit_excit_setVars(imfilename, pnevFileName, sigTh, goodinds, showResults);
    
    
    %% set traces for good inhibit and excit neurons.
    
    alldataDfofGoodInh = cellfun(@(x)x(:, good_inhibit), alldataDfofGood, 'uniformoutput', 0); % 1 x number of trials
    alldataSpikesGoodInh = cellfun(@(x)x(:, good_inhibit), alldataSpikesGood, 'uniformoutput', 0); % 1 x number of trials
    
    alldataDfofGoodExc = cellfun(@(x)x(:, good_excit), alldataDfofGood, 'uniformoutput', 0); % 1 x number of trials
    alldataSpikesGoodExc = cellfun(@(x)x(:, good_excit), alldataSpikesGood, 'uniformoutput', 0); % 1 x number of trials
    
    % for the rest its easy, just to (:, good_inhibit) and (:, good_excit) to
    % get their corresponding traces for inhibit and excit neurons.
    % activityGood
    % dfofGood
    % spikesGood
    % alldataDfofGood_mat
    
end





%%
%%%%%%%%%%%%%%%%%%%%%%%%% Do some plotting of neural traces %%%%%%%%%%%%%%%%%%%%%%%

%% Look at traces per neuron and per trial and show events during a trial as well. Traces start at the scan start (so not aligned on any particular event.)

if plotTraces1by1
    
    % it makes sense to look at all trials and not just ~trs2rmv so do
    % trs2rmv = unique([trs_problemAlign, trs_badMotion_pmtOff]);, and again run setEvent
    plotTrs1by1 = 1;
    interactZoom = 0; 
    markQuantPeaks = 1;
    allEventTimes = {timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, timeStop, centerLicks, leftLicks, rightLicks};
    [~, ~, stimrate] = setStimRateType(alldata);

    % plotCaTracesPerNeuron([{C_df'}, {C_mcmc_df'}], interactZoom, plotTrs1by1, markQuantPeaks, alldataDfofGood, alldataSpikesGood, {framesPerTrial, alldata, S_mcmc', dFOF_man, matchedROI_idx, allEventTimes, stimrate})
    plotCaTracesPerNeuron({dfofGood}, interactZoom, plotTrs1by1, markQuantPeaks, alldataDfofGood, alldataSpikesGood, {framesPerTrial, alldata, spikesGood, [], [], allEventTimes, stimrate})

    % plotCaTracesPerNeuron(traceFU_toplot, interactZoom, plotTrs1by1, markQuantPeaks, {framesPerTrial, alldata, S_mcmc, dFOF_man, matchedROI_idx})


    %% plot all neuron traces per trial

    plotCaTracesPerTrial

end


%% Plot average traces across all neurons and all trials aligned on particular trial events. 

avetrialAlign_plotAve_noTrGroup


%% Plot average traces across all neurons for different trial groups aligned on particular trial events. 

fprintf('Remember: in each subplot (ie each neuron) different trials are contributing to different segments (alignments) of the trace!\n')
avetrialAlign_plotAve_trGroup






%%
%%%%%%%%%%%%%%%%%%%%%%%%% Start some analyses: alignement, choice preference, SVM %%%%%%%%%%%%%%%%%%%%%%%


%% Align traces on particular trial events

% remember traces_al_sm has nan for trs2rmv as well as trs in alignedEvent that are nan.

traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
alignedEvent = 'stimOn'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
dofilter = true; % false; 
% set nPre and nPost to nan if you want to go with the numbers that are based on eventBef and eventAft.
% set to [] to include all frames before and after the alignedEvent.
nPreFrames = nan; []; % nan; 
nPostFrames = nan; []; % nan;

traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.

[traces_al_sm, time_aligned_stimOn, eventI_stimOn] = alignTraces_prePost_filt(traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, nPreFrames, nPostFrames);

% set to nan those trials in outcomes and allRes that are nan in traces_al_sm
a = find(sum(sum(~isnan(traces_al_sm),1),3), 1);
allTrs2rmv = find(squeeze(sum(isnan(traces_al_sm(:,a,:)))));
outcomes(allTrs2rmv) = NaN; 
allResp(allTrs2rmv) = NaN; 
allResp_HR_LR(allTrs2rmv) = NaN;


%% Compute choice preference, 2*(auc-0.5), for each neuron at each frame.

% choicePref_ROC

% set ipsi and contra trials
thStimStrength = 2; % 2; % what stim strength you want to use for computing choice pref.

correctL = (outcomes==1) & (allResp==1);
correctR = (outcomes==1) & (allResp==2);

ipsiTrs = (correctL' &  abs(stimrate-cb) > thStimStrength);
contraTrs = (correctR' &  abs(stimrate-cb) > thStimStrength);
fprintf('Num corrL and corrR (stim diff > %d): %d,  %d\n', [thStimStrength, sum(ipsiTrs) sum(contraTrs)])

makeplots = 1;
useEqualNumTrs = false; % if true, equal number of trials for HR and LR will be used to compute ROC.

% compute choicePref for each frame
% choicePref_all: frames x units. chiocePref at each frame for each neuron
choicePref_all = choicePref_ROC(traces_al_sm, ipsiTrs, contraTrs, makeplots, eventI_stimOn, useEqualNumTrs);


%% Compute choicePref for the average of frames after the stim.

traces_al_sm_aveFr = nanmean(traces_al_sm(eventI_stimOn:end,:,:), 1);
choicePref_all = choicePref_ROC(traces_al_sm_aveFr, ipsiTrs, contraTrs, makeplots, eventI_stimOn, useEqualNumTrs);


%% SVM

% do the analysis
% popClassifier
popClassifier_GE

% plot projections
popClassifierSVM_plots

% for CV dataset
popClassifierSVM_plots_CVprojections.m

% Compare svm weights with random weights
popClassifierSVM_rand

% Compare SVM weights with ROC choicePref
popClassifierSVM_choicePref











%% old stuff
%{
%%
% framesPerTr = framesPerTrial(~isnan(framesPerTrial));
% csfrs = [0 cumsum(framesPerTr)];

% Note: you are calling decisions that happen during lickcenter_again
% also early decision which is not quite correct, bc they happened
% after the stim was over. A better name would be
% lick_center_again_early_decision


%% Set the baseline for each trial and each neuron
bl = NaN(size(traceEventAlign, 2), size(traceEventAlign, 3)); % neurons x trials

for itr = 1:size(bl,2)
    for ineu = 1:size(bl,1)
        if ~isnan(timeInitTone{itr})            
            f = eventFrameNum-1;
%             blRange = 1:f; % all frames preceding toneInit
            blRange = f-floor(100/frameLength) : f; % 100ms preceding toneInit
            bl(ineu,itr) = nanmean(traceEventAlign(blRange,ineu,itr));
        end
    end
end


%% Set to NaN trials with unstable baselines. Remember that trs2rmv are already set to NaN.
bl_th = 1.5; % bl_stable*5; % *5 needs assessment.
figure; plot(sum(bl>bl_th, 2)) % number of unstable bl trials for each neuron.
traceEventAlign_stableBl = traceEventAlign;
traceEventAlign_stableBl(:, bl>bl_th) = NaN;

traceAveTrs = nanmean(traceEventAlign_stableBl,3); % frames x neurons, average across trials.

nvalidtrs = sum(~isnan(traceEventAlign_stableBl),3); % frames x neurons; number of trials that contribute to each frame for each neuron.
figure; plot(nvalidtrs)


%% set to NaN those frames to which only a few trials contribute.
thNumTr = 15;
% traceEventAlign_stableBl(nvalidtrs<thNumTr) = NaN;
traceAveTrs(nvalidtrs<thNumTr) = NaN;

%% Look at average trials per neuron
figure; 
for ineu = 1:size(traceAveTrs,2)
    set(gcf, 'name', sprintf('Neuron %d', ineu))
    hold on
    toplot = traceAveTrs(:,ineu);
    plot(toplot)
    
    mn = min(toplot(:));
    mx = max(toplot(:));
    plot([eventFrameNum eventFrameNum],[mn mx],'k:')
    
    pause
    delete(gca)
end


%% ave across all trials per neuron
% thNumTr = 0;

alldata(1).highRateChoicePort
group1 = lowRateTrials;
group2 = highRateTrials;

% group1 = leftChoiceTrials;
% group2 = rightChoiceTrials;

group1_aveTrace = nanmean(traceEventAlign(:,:,group1), 3)'; % frames x units. Average across trials.
group2_aveTrace = nanmean(traceEventAlign(:,:,group2), 3)'; 

% number of trials that contribute to each frame for each neuron. (it
% is obviously the same for all neurons, bc they were recorded
% simultaneously).
ss1 = sum(~isnan(traceEventAlign(:,:,group1)),3)'; % frames x units. sum across trials.
group1_aveTrace(ss1<thNumTr) = NaN;

ss2 = sum(~isnan(traceEventAlign(:,:,group2)),3)';
group2_aveTrace(ss2<thNumTr) = NaN;

figure; hold on, plot(ss1(1,:)), plot(ss2(1,:))
xlabel('Frame number')
ylabel('Number of trials contributing to the average trace')

% N 7 clear tuning for the stim.


%% ave across all neurons per trial
group1 = lowRateTrials;
group2 = highRateTrials;

% group1 = leftChoiceTrials;
% group2 = rightChoiceTrials;

group1_aveTrace = squeeze(nanmean(traceEventAlign(:,:,group1), 2)); 
group2_aveTrace = squeeze(nanmean(traceEventAlign(:,:,group2), 2)); 

figure; hold on
plot(timeEventAlign, nanmean(group1_aveTrace,2))
plot(timeEventAlign, nanmean(group2_aveTrace,2))
plot([0 0],[-1 2], 'color', [.6 .6 .6])
xlim([-5000 5000])
ylim([0 1])

pause
for n = 1:max([sum(rightChoiceTrials) sum(leftChoiceTrials)])
    hl = plot(timeEventAlign, group1_aveTrace(:,n), 'color', [.6 .6 .6]);
    hh = plot(timeEventAlign, group2_aveTrace(:,n), 'k');
    title(n)
    pause
    delete([hl hh])
end





%%
na = 1:numUnits;
ngood = na(traceQuality==qual2an);

n = 1;
tr = 1;

figure; hold on
f = traceEventAlign(:,n,tr);
plot(timeEventAlign, f)
plot(alldata(tr).frameTimes, alldata(tr).dFOF(:,ngood(n)))

%% ave trials per neuron
f = squeeze(nanmean(traceEventAlign,3));

figure; hold on
plot(timeEventAlign, nanmean(f,2))
% ylim([0 .6])
xlim([-1000 5000])

for n = 1:size(f,2)
    h = plot(timeEventAlign, f(:,n));
    pause
    delete(h)
end


%% ave neurons per trial
f = squeeze(nanmean(traceEventAlign,2));

figure; hold on
plot(timeEventAlign, nanmean(f,2))
ylim([0 .6])
xlim([-1000 5000])

for n = 1:size(f,2)
    h = plot(timeEventAlign, f(:,n));
    pause
    delete(h)
end



%%
na = 1:numUnits;
ngood = na(traceQuality==1);

p = get(0,'screensize');
figure('position',[1, p(end)-200, p(3), 120])
plot(dfof(:,ngood))

qual_best = NaN(1,numUnits);
for n = ngood
    plot(dfof(:,n))
    ylim([-.5 1.5])
    xlim([0 length(dfof)])
    qual_best(n) = input('great? ');
end

%%
plot_alltraces(dfof(:,qual_best==1)', 'mf', 4);




%%
%{
% remmber bcontrol states are in sec.
%{

                        state_0: [NaN 2.6015e+03]
         check_next_trial_ready: [0x2 double]
                            iti: [14x2 double]
                           iti2: [13x2 double]
             start_rotary_scope: [3x2 double]
            start_rotary_scope2: [3x2 double]
                     trialcode1: [2.6080e+03 2.6080e+03]
                  trialcode1low: [2.6080e+03 2.6080e+03]
                     trialcode2: [2.6080e+03 2.6080e+03]
                  trialcode2low: [2.6080e+03 2.6080e+03]
                     trialcode3: [2.6080e+03 2.6080e+03]
                  trialcode3low: [2.6080e+03 2.6080e+03]
                     trialcode4: [2.6080e+03 2.6080e+03]
                  trialcode4low: [2.6080e+03 2.6080e+03]
                     trialcode5: [2.6080e+03 2.6080e+03]
                  trialcode5low: [2.6080e+03 2.6080e+03]
                     trialcode6: [2.6080e+03 2.6080e+03]
                  trialcode6low: [2.6080e+03 2.6080e+03]
                     trialcode7: [2.6080e+03 2.6080e+03]
                  trialcode7low: [2.6080e+03 2.6080e+03]
                     trialcode8: [2.6080e+03 2.6080e+03]
                  trialcode8low: [2.6080e+03 2.6080e+03]
                     trialcode9: [2.6080e+03 2.6080e+03]
                  trialcode9low: [2.6080e+03 2.6080e+03]
            wait_for_initiation: [2.6080e+03 2.6081e+03]
           wait_for_initiation2: [0x2 double]
                     stim_delay: [2.6081e+03 2.6081e+03]
                      wait_stim: [2.6081e+03 2.6082e+03]
               lickcenter_again: [2.6082e+03 2.6082e+03]
                  center_reward: [2.6082e+03 2.6082e+03]
                   direct_water: [0x2 double]
              wait_for_decision: [2.6082e+03 2.6086e+03]
         correctlick_again_wait: [2.6086e+03 2.6087e+03]
           errorlick_again_wait: [0x2 double]
              correctlick_again: [2.6087e+03 2.6089e+03]
                errorlick_again: [0x2 double]
             wait_for_decision2: [0x2 double]
               wrong_initiation: [0x2 double]
                early_decision0: [0x2 double]
                 early_decision: [0x2 double]
              did_not_lickagain: [0x2 double]
                 did_not_choose: [0x2 double]
          did_not_sidelickagain: [0x2 double]
                 direct_correct: [0x2 double]
                         reward: [2.6089e+03 2.6090e+03]
                   stopstim_pre: [2.6090e+03 2.6100e+03]
                reward_stopstim: [2.6100e+03 2.6100e+03]
         punish_allowcorrection: [0x2 double]
    punish_allowcorrection_done: [0x2 double]
                         punish: [0x2 double]
                 punish_timeout: [0x2 double]
              stop_rotary_scope: [2.6100e+03 NaN]
                 starting_state: 'state_0'
                   ending_state: 'stop_rotary_scope'
                   
                   
%}

%%
% duration of the session in minutes:
(alldata(end-1).parsedEvents.states.stop_rotary_scope(1) - alldata(1).parsedEvents.states.state_0(2)) / 64

% the following are equal:
alldata(tr).frameTimes(1) - (1000/30.9)/2 - (1000*diff(alldata(tr).parsedEvents.states.start_rotary_scope(1,:)))
frame1RelToStartOff(tr)



%%
% trial initiation time (when mouse did the 1st lick)
alldata(tr).parsedEvents.states.stim_delay(1)

% commit center lick , center reward , go tone.
alldata(tr).parsedEvents.states.center_reward(1)

% commit correct lick , reward
alldata(tr).parsedEvents.states.reward(1)

% remember 1st, 2nd, correction, etc... for analysis.


%% all times are relative to scan start except those whose name ends at _abx
frameLength = 1000/30.9; % sec.

% duration (ms) that MScan lagged behind in executing bcontrol command signal
mscanLagRelBcontrol = alldata(tr).frameTimes(1) - frameLength/2;

% _abs indicates that the time is relative to the beginning of the session
% (not the begining of the trial).
timeScopeTTLsent_abs = alldata(tr).parsedEvents.states.start_rotary_scope(1,1) * 1000;

% this is what you need to subtract from the timepoints in a trial to find
% how far they are from the scanning origin.
timeStartScan_abs = timeScopeTTLsent_abs + mscanLagRelBcontrol; % in ms (time 0 is beginning of the session).

% trial initiation time (when mouse did the 1st lick), relative to the
% begining of scanning.
time1stCenterLick = alldata(tr).parsedEvents.states.stim_delay(1) * 1000 - timeStartScan_abs

% commit center lick , center reward , go tone, relative to scan start.
timeCommitCL_CR_Gotone = alldata(tr).parsedEvents.states.center_reward(1) * 1000 - timeStartScan_abs

% commit correct lick , reward, relative to scan start.
timeReward = alldata(tr).parsedEvents.states.reward(1) * 1000 - timeStartScan_abs

% first side response, relative to scan start.
time1stSideResponse = min([alldata(tr).parsedEvents.states.correctlick_again_wait(:,1);...
alldata(tr).parsedEvents.states.errorlick_again_wait(:,1)]) * 1000 - timeStartScan_abs


% all center reponses, relative to scan start.
timeAllCenterResp = alldata(tr).parsedEvents.pokes.C(:,1) * 1000 - timeStartScan_abs

% all left reponses, relative to scan start.
timeAllLeftResp = alldata(tr).parsedEvents.pokes.L(:,1) * 1000 - timeStartScan_abs

% all right reponses, relative to scan start.
timeAllRightResp = alldata(tr).parsedEvents.pokes.R(:,1) * 1000 - timeStartScan_abs

%}


%% duration of no scopeTTL, preceding the trial
dur_nottl = NaN(1,length(all_data)-1);
for tr = 2:length(all_data)-1
    dur_nottl(tr) = all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000 - all_data(tr-1).parsedEvents.states.stop_rotary_scope(1)*1000;
end
[m, i] = min(dur_nottl)


%% duration of a trial in mscan (ie duration of scopeTTL being sent).
frameLength = 1000 / 30.9;
nfrs = NaN(1,length(all_data)-1);
for tr = 1:length(all_data)-1
    durtr = all_data(tr).parsedEvents.states.stop_rotary_scope(1)*1000 + 500 - ...
        all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000; % 500 is added bc the duration of stop_rotary_scope is 500ms.
    
    nfrs(tr) = durtr/ frameLength;
end
size(nfrs)

% duration of state check_next_trial_ready
% all_data(tr+1).parsedEvents.states.state_0(2)*1000 - all_data(tr).parsedEvents.states.stop_rotary_scope(1)*1000


%%
% trials that miss a trial code (ie were not recorded in mscan but recorded
% in bcontrol)
trmiss = find(~ismember(1:length(trialNumbers), trialNumbers))
nfrs(trmiss) = [];

%%
size(cell2mat(framesPerTrial)')
f = cell2mat(framesPerTrial)';

[(1:length(framecountFrames))' nfrs' framecountFrames' f(1:length(framecountFrames))]

[(1:length(f))' nfrs(1:length(f))' framecountFrames(1:length(f))' f]

% [(1:length(framecountFrames))' framecountFrames' framesPerTrial(1:length(framecountFrames))']



%% Checks
%{
% the following will be equal
tr = 1;
isequal(-1000 *alldata(tr).parsedEvents.states.center_reward(1) + (alldata(tr).parsedEvents.states.stim_delay(1)*1000) , ...
    (time1stCenterLick(tr) - timeCommitCL_CR_Gotone(tr)))
%}
    
%{
[sum(~isnan(time1stCenterLick)),...
sum(~isnan(timeStimOnset)),...
sum(~isnan(timeCommitCL_CR_Gotone)),...
sum(~isnan(time1stSideResponse)),...
sum(~isnan(timeReward)),...
sum(~isnan(time1stCorrectResponse))]

figure; hold on
plot(time1stCenterLick)
plot(timeStimOnset)
plot(timeCommitCL_CR_Gotone)
plot(time1stSideResponse)
plot(timeReward)
%}



%}