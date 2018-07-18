%{

% clear; close all
mouse = 'fni16'; %'fni17';
imagingFolder = '151028'; %'151015';
mdfFileNumber = [1,2];

%%
% close all
% best is to set the 2 vars below to 0 so u get times of events for all trials; later decide which ones to set to nan.
rmvTrsStimRateChanged = 0; % if 1, early-go-tone trials w stimRate categ different before and after go tone, will be excluded.
do = 0; % set to 1 when first evaluating a session, to get plots and save figs and vars.

normalizeSpikes = 1; % if 1, spikes trace of each neuron will be normalized by its max.
warning('Note you have set normalizeSpikes to 1!!!')

thbeg = 5; % n initial trials to exclude.
if strcmp(mouse, 'fni19') && strcmp(imagingFolder,'150918')
    thbeg = 7;
end
% normally rmvTrsStimRateChanged is 1, except for early sessions of
% training we have to set it to 0, otherwise lots of trials will be
% excluded bc go tone has happpened very early.

rmv_timeGoTone_if_stimOffset_aft_goTone = 0; % if 1, trials with stimOffset after goTone will be removed from timeGoTone (ie any analyses that aligns trials on the go tone)
rmv_time1stSide_if_stimOffset_aft_1stSide = 0; % if 1, trials with stimOffset after 1stSideTry will be removed from time1stSideTry (ie any analyses that aligns trials on the 1stSideTry)

evaluateEftyOuts = do; 
compareManual = do; % compare results with manual ROI extraction
plot_ave_noTrGroup = do; % Set to 1 when analyzing a session for the 1st time. Plots average imaging traces across all neurons and all trials aligned on particular trial events. Also plots average lick traces aligned on trial events.
save_aligned_traces = do; % save aligned_traces to postName
savefigs = do;

setInhibitExcit = 0; % if 1, inhibitory and excitatory neurons will be set unless inhibitRois is already saved in imfilename (in which case it will be loaded).
plotEftyAC1by1 = 0; % A and C for each component will be plotted 1 by 1 for evaluation of of Efty's results. 
frameLength = 1000/30.9; % sec.

[alldata, alldataSpikesGood, alldataDfofGood, goodinds, good_excit, good_inhibit, outcomes, allResp, allResp_HR_LR, ...
        trs2rmv, stimdur, stimrate, stimtype, cb, timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, ...
        timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, imfilename, pnevFileName] ....
   = wheelRevAlign(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength, save_aligned_traces, savefigs, rmvTrsStimRateChanged, thbeg);


%}

%%
function [alldata, alldataSpikesGood, alldataDfofGood, goodinds, good_excit, good_inhibit, outcomes, allResp, allResp_HR_LR, ...
        trs2rmv, stimdur, stimrate, stimtype, cb, timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, ...
        timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, imfilename, pnevFileName] ....
        = tracesAlign_wheelRev_lick(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength, save_aligned_traces, savefigs, rmvTrsStimRateChanged, thbeg);

% Right after you are done with preproc on the cluster, run the following scripts:
% - plotEftyVarsMean (if needed follow by setPmtOffFrames to set pmtOffFrames and by findTrsWithMissingFrames to set frame-dropped trials. In this latter case you will need to rerun CNMF!): for a quick evaluation of the traces and spotting any potential frame drops, etc
% - eval_comp_main on python (to save outputs of Andrea's evaluation of components in a mat file named more_pnevFile)
% - set_mask_CC
% - findBadROIs
% - inhibit_excit_prep
% - imaging_prep_analysis (calls set_aligned_traces... you will need its outputs)
%
% This is the main and starting function for the analysis of your imaging data. It gives you the vars that you need for further analyses. It used to be named aveTrialAlign_setVars
%
% pnev_manual_comp_setVars is also a very nice script (together with pnev_manual_comp_match) that allows you to plot and compare the trace and ROIs of 2 different methods.
% You can use it to compare Eftychios vs manual. Or 2 different channels. Or 2 different methods of Eftychios, etc.
%
% worry about bleedthrough, and visual artifact.... it seems not to be a concern when using Eftychios's algorithm:
%
% outcomes:
%    1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation,
%   -4: no center commit, -5: no side commit
%
% Farzaneh Najafi (2016)
%
% Example input variales:
%{
mouse = 'fni17';
imagingFolder = '151020'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1,2];  % 3; %1; % or tif major

% best is to set the 2 vars below to 0 so u get times of events for all trials; later decide which ones to set to nan.
rmv_timeGoTone_if_stimOffset_aft_goTone = 0; % if 1, trials with stimOffset after goTone will be removed from timeGoTone (ie any analyses that aligns trials on the go tone)
rmv_time1stSide_if_stimOffset_aft_1stSide = 0; % if 1, trials with stimOffset after 1stSideTry will be removed from time1stSideTry (ie any analyses that aligns trials on the 1stSideTry)

normalizeSpikes = 1; % if 1, spikes trace of each neuron will be normalized by its max.

% set the following vars to 1 when first evaluating a session.
evaluateEftyOuts = 0; 
compareManual = 0; % compare results with manual ROI extraction
plot_ave_noTrGroup = 0; % Set to 1 when analyzing a session for the 1st time. Plots average imaging traces across all neurons and all trials aligned on particular trial events. Also plots average lick traces aligned on trial events.
save_aligned_traces = 0;
savefigs = 0;

setInhibitExcit = 0; % if 1, inhibitory and excitatory neurons will be set unless inhibitRois is already saved in imfilename (in which case it will be loaded).
plotEftyAC1by1 = 0; % A and C for each component will be plotted 1 by 1 for evaluation of of Efty's results. 
frameLength = 1000/30.9; % sec.

[alldata, alldataSpikesGood, alldataDfofGood, goodinds, good_excit, good_inhibit, outcomes, allResp, allResp_HR_LR, ...
        trs2rmv, stimdur, stimrate, stimtype, cb, timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, ...
        timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, imfilename, pnevFileName] = ....
   imaging_prep_analysis(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength);


% Once done use set_aligned_traces to set aligned traces on different trial events with carefully chosen trials.
%{
evaluateEftyOuts = 0; 
compareManual = 0; % compare results with manual ROI extraction
plot_ave_noTrGroup = 0; 
%}
%}

home

% Rmemeber once preproc is done, you need to run eval_comp_main in Python
% to save outputs of Andrea's evaluation of components in a mat file named
% more_pnevFile... Then run set_mask_CC to append to this mat file, A and CC.


%% Good days
% imagingFolder = '151029'; % '151021';
% mdfFileNumber = 3;


%% Set some initial variables.

    %{
    assessInhibitClass = 0; % if 1 and inhibitRois not already saved, you will evaluate identification of inhibitory neurons (ie if sigTh is doing a good job).
    saveInhibitRois = 0; % if 1 and inhibitRois not already saved, ROIs that were identified as inhibit will be saved in imfilename. (ie the output of inhibit_excit_setVars will be saved).
    quantTh = .8; % quantile of roi_sig2surr that will be used for inhibit identification. 1.13; % signal to noise threshold for identifying inhibitory neurons on tdtomato channel. eg. sigTh = 1.2;
    % 1.1: excit is safe, but u should check inhibit with low sig/surr to make sure they are not excit.
    % 1.2: u are perhaps missing some inhibit neurons.
    % 1.13 can be good too.
    %}
    
excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;
%{
allowCorrectResp = 'change'; % 'change'; 'remove'; 'nothing'; % if 'change': on trials that mouse corrected his choice, go with the original response.
uncommittedResp = 'nothing'; % 'change'; 'remove'; 'nothing'; % what to do on trials that mouse made a response (licked the side port) but did not lick again to commit it.

plot_ave_trGroup = 0; % Plot average traces across all neurons for different trial groups aligned on particular trial events.
plotTrialTraces1by1 = 0; % plot traces per neuron and per trial showing all trial events
furtherAnalyses = 0; % analyses related to choicePref and SVM will be performed.

setNaN_goToneEarlierThanStimOffset = 0; % if 1, set to nan eventTimes of trials that had go tone earlier than stim offset... if 0, only goTone time will be set to nan provided that rmv_timeGoTone_if_stimOffset_aft_goTone = 1
%}
%{
autoTraceQual = 0; %1; % if 1, automatic measure for trace quality will be used.
manualExamineTraceQual = 0; % if 0, traceQuality array needs to be saved.
    saveTraceQual = 0; % it will only take effect if manualExamineTraceQual is 1.
    analyzeQuality = [1 2]; % 1(good) 2(ok-good) 3(ok-bad) 4(bad) % trace qualities that will be analyzed. It will only take effect if manualExamineTraceQual is 1.
orderTraces = 0; % if 1, traces will be ordered based on the measure of quality from high to low quality.
%}

if ~exist('thbeg', 'var')
    thbeg = 5; % n initial trials to exclude.
end
signalCh = 2;
pnev2load = []; %7 %4 % what pnev file to load (index based on sort from the latest pnev vile). Set [] to load the latest one.


%{
helpedInit = []; % [100];
helpedChoice = []; %31;
defaultHelpedTrs = 0; % if 1, the program assumes that no trial was helped.
saveHelpedTrs = 0; % it will only take effect if defaultHelpedTrs is false. If 1, helpedTr fields will be added to alldata.
%}
% analyzeOutcomes = {'all'}; % {'success', 'failure'}; % outcomes that will be analyzed.
% outName = 'fni17-151016';


%%

% set the names of imaging-related .mat file names.
% remember the last saved pnev mat file will be the pnevFileName
%{
signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
%}
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
% [~,b] = fileparts(imfilename);
% diary(['diary_',b])
disp(pnev_n)
cd(fileparts(imfilename))

moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
aimf = matfile(imfilename);

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

hrChoiceSide = all_data(1).highRateChoicePort;


% begTrs = [0 cumsum(trials_per_session)]+1;
% begTrs = begTrs(1:end-1);
begTrs = 1; % 1st trial of each session

load(imfilename, 'outputsDFT', 'badFrames', 'pmtOffFrames')

minPts = 7000; %800;
smoothPts = 6;


%%
%%%%%%%%%%%%%%%%%%%%%%%%% Load neural data, merge them into all_data, and assess trace quality %%%%%%%%%%%%%%%%%%%%%%%

spikes = nan(length(badFrames{1}),2); % we just need it so merging (below) doesn't give error! % assume we have 2 neruons
activity = spikes;
dFOF = spikes;
goodinds = ones(1,2);


%%

[nFrsSess, nFrsMov] = set_nFrsSess(mouse, imagingFolder, mdfFileNumber); % nFrsMov: for each session shows the number of frames in each tif file.
cs_frmovs = [0, cumsum(cell2mat(nFrsMov))]; % cumsum of nFrsMov: shows number of frames per tif movie (includes all tif movies of mdfFileNumber). 


%% Load vars related to manual method

%{
load(pnevFileName, 'activity_man_eftMask_ch2')

spikes = activity_man_eftMask_ch2;
activity = activity_man_eftMask_ch2; %temporalComp'; % frames x units % temporalComp = C; % C_mcmc; % C2; % obj.C;
dFOF = activity_man_eftMask_ch2; %konnerthDeltaFOverF(activity_man_eftMask_ch2, pmtOffFrames{1}, smoothPts, minPts);


% if compareManual
    
    load(imfilename, 'imHeight', 'imWidth', 'pmtOffFrames')
    
    %{
    load(imfilename, 'rois', 'activity') % , 'imHeight', 'imWidth', 'pmtOffFrames')
    load(pnevFileName, 'A') % load('demo_results_fni17-151102_001.mat', 'A2')
    spatialComp = A; % A2 % obj.A;
    clear A
    %}
    
    %% manual activity and dfof
%     activity_man = activity_man_eftMask_ch2;    
    %     activity_man = activity;
%     clear activity activity_man_eftMask*
    
    % Compute Df/f for the manually found activity trace.
    gcampCh = 2; smoothPts = 6; minPts = 7000; %800;
    
    % in case there are all-nan traces (can happen if ROI is too small), remove them before runing
    % konnarthDeltaF because runningF doesnt accept nans. 
    a = sum(isnan(activity_man_eftMask_ch2));     b = sum(a,1);
    allNaNrois = find(b);
    if ~isempty(allNaNrois)
        cprintf('red', 'ROI %d has all-nan manAct!\n', allNaNrois)
        activity_man_eftMask_ch2(:,allNaNrois) = [];
    end
    
    dFOF_man = konnerthDeltaFOverF(activity_man_eftMask_ch2, pmtOffFrames{gcampCh}, smoothPts, minPts);
    
    if ~isempty(allNaNrois)
        activity_man_eftMask_ch2 = insertElement(activity_man_eftMask_ch2', allNaNrois, nan); 
        activity_man_eftMask_ch2 =  activity_man_eftMask_ch2';
        dFOF_man = insertElement(dFOF_man', allNaNrois, nan);
        dFOF_man = dFOF_man';
    end
    
    
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
    
% end


%% Assess pixel shifts (ie output of motion correction), pmtOffFrames and badFrames (frames with too much motion)

load(imfilename, 'outputsDFT', 'badFrames', 'pmtOffFrames')


%% Load and clean Efty's vars
%{
signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
disp(pnev_n)
cd(fileparts(imfilename))

normalizeSpikes = 1;
load(pnevFileName, 'activity_man_eftMask_ch2')
load(imfilename, 'cs_frtrs')
load(pnevFileName, 'activity_man_eftMask_ch1')
%}
fprintf('Loading Eftys vars...')
% load(pnevFileName, 'C', 'C_df', 'S', 'A', 'P', 'f')
load(pnevFileName, 'C', 'C_df', 'S', 'f')
load(pnevFileName, 'A', 'P')
fprintf('...done\n')

load(imfilename, 'Nnan_nanBeg_nanEnd')
% S(:, [32672       32333       32439       32547]) = nan; % sharp spikes due to frame missing (their trials will be excluded... you are just doing this so they dont affect the normalization.)
[C, S, C_df] = processEftyOuts(C, S, C_df, Nnan_nanBeg_nanEnd, normalizeSpikes);

% set time constants (in ms) from P.gn
% frameLength = 1000/30.9; % sec.
tau = nan(size(P.gn,1), 2);
for i = 1:length(tau)
    g = P.gn{i};
    tau(i,:) = tau_d2c(g,frameLength); % tau(:,1) is rise, and tau(:,2) is decay time constant (in ms).
end


%% Set number of frames per session

% if length(mdfFileNumber)>1
[nFrsSess, nFrsMov] = set_nFrsSess(mouse, imagingFolder, mdfFileNumber); % nFrsMov: for each session shows the number of frames in each tif file.
cs_frmovs = [0, cumsum(cell2mat(nFrsMov))]; % cumsum of nFrsMov: shows number of frames per tif movie (includes all tif movies of mdfFileNumber). 
% frs = cs_frmovs(itif)+1 : cs_frmovs(itif+1); % frames that belong to movie itif (indeces corresponds to the entire movie) 
a = matfile(imfilename);
if ~isprop(a, 'nFrsMov')
    save(imfilename, '-append', 'nFrsMov', 'cs_frmovs')  
end


%%
spikes = S';
activity = C'; %temporalComp'; % frames x units % temporalComp = C; % C_mcmc; % C2; % obj.C;
% activity = activity_man_eftMask_ch2;
% dFOF = C_df'; % temporalDf'; temporalDf = C_df; % C_mcmc_df; % C_df; % obj.C_df;
minPts = 7000; %800;
smoothPts = 6;
dFOF = dFOF_man; %konnerthDeltaFOverF(activity_man_eftMask_ch2, pmtOffFrames{1}, smoothPts, minPts);


%{
spikes = C';
activity = activity_man_eftMask_ch2;
dFOF = []; % to use Konnath df/f
%}
clear C C_df S dFOF_man
%}

%%
%%%%%%%%%%%%%%%%%%%%%%%%% Merging imaging data to behavioral data %%%%%%%%%%%%%%%%%%%%%%%

if length(mdfFileNumber)==1
    
    load(imfilename, 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'trialCodeMissing') %, 'cs_frtrs')
   
%     figure('position', [68   331   254   139]); hold on; plot(framesPerTrial - diff(cs_frtrs))
%     xlabel('trial'), ylabel('framesPerTrial - diff(cs_frtrs)'), title('shows frame-dopped trials')

    
    if ~exist('trialCodeMissing', 'var'), trialCodeMissing = []; end  % for those days that you didn't save this var.
    
    if length(trialNumbers) ~= length(framesPerTrial)
        error('Investigate this. Make sure merging with imaging is done correctly.')
    end
    
    trialNumbers(isnan(framesPerTrial)) = [];
    frame1RelToStartOff(isnan(framesPerTrial)) = [];
    framesPerTrial(isnan(framesPerTrial)) = [];    
    
    
    cprintf('blue', 'Total number of imaged trials: %d\n', length(trialNumbers))
    if ~any(trialCodeMissing)
        cprintf('blue', 'All trials are triggered in MScan :)\n')
    else
        cprintf('blue', ['There are non-triggered trials in MScan! Trial(s):', repmat('%d ', 1, sum(trialCodeMissing)), '\n'], find(trialCodeMissing))
    end
    
    
    %% Remove trials at the end of alldata that are in behavior but not imaging.
    
    max_num_imaged_trs = max(length(framesPerTrial), max(trialNumbers)); % I think you should just do : max(trialNumbers)
    
    a = length(all_data) - max_num_imaged_trs;
    if a~=0
        cprintf('blue', 'Removing %i trials at the end of alldata bc imaging was aborted.\n', a)
    else
        fprintf('Removing no trials at the end of alldata bc imaging was not aborted earlier.\n')
    end
    
    all_data(max_num_imaged_trs+1: end) = []; % remove behavioral trials at the end that with no recording of imaging data.
    
    
    %% Merge imaging variables into all_data (before removing any trials): activity, dFOF, spikes

    minPts = 7000; %800;
    set2nan = 1; % if 1, in trials that were not imaged, set frameTimes, dFOF, spikes and activity traces to all nans (size: min(framesPerTrial) x #neurons).
    
    [all_data, mscanLag] = mergeActivityIntoAlldata_fn(all_data, activity, framesPerTrial, ...
        trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, dFOF, spikes, set2nan);
    
    alldata = all_data;
    
    
    % manual
    % activity = activity_man;
    % dFOF = dFOF_man;
    % spikes = dFOF_man;
    % clear dFOF_man activity_man
    % [all_data, mscanLag] = mergeActivityIntoAlldata_fn(all_data, activity, framesPerTrial, ...
    %   trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, dFOF, spikes);
    
        
    %{
    if sum(~ismember(1:length(trialNumbers), trialNumbers))~=0
        error('There are non-triggered trials in mscan! Decide how to proceed!')
        % try this:
        % remove the trial that was not recorded in MScan
        all_data(~ismember(1:length(trialNumbers), trialNumbers)) = []; % FN note: double check this.
    end
    alldata = all_data(trialNumbers); % in case mscan crashed, you want to remove trials that were recorded in bcontrol but not in mscan.
    %}    
    
    
    %% Set trs2rmv, stimrate, outcome and response side. You will set certain variables to NaN for trs2rmv (but you will never remove them from any arrays).
    
    load(imfilename, 'badAlignTrStartCode', 'trialStartMissing'); %, 'trialCodeMissing') % they get set in framesPerTrialStopStart3An_fn
    load(imfilename, 'trEndMissing', 'trEndMissingUnknown', 'trStartMissingUnknown') % Remember their indeces are on the imaged trials (not alldata). % do trialNumbers(trEndMissing) to find the corresponding indeces on alldata. 
    
    if ~exist('trEndMissing', 'var'), trEndMissing = []; end
    if ~exist('trEndMissingUnknown', 'var'), trEndMissingUnknown = []; end
    if ~exist('trStartMissingUnknown', 'var'), trStartMissingUnknown = []; end
    
    imagingFlg = 1;
    [trs2rmv, stimdur, stimrate, stimtype, cb] = setTrs2rmv_final(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, imagingFlg, badAlignTrStartCode, trialStartMissing, trialCodeMissing, trStartMissingUnknown, trEndMissing, trEndMissingUnknown, trialNumbers, rmvTrsStimRateChanged);
    trs2rmv(trs2rmv>length(alldata)) = [];
    
else
    
    % assing pmtOff and badFrames to each session
    % merge alldata with imaging for each session
    % set trs2rmv for each session
    multi_sess_set_vars
    
    if ~isprop(aimf, 'len_alldata_eachsess')
        save(imfilename, '-append', 'len_alldata_eachsess', 'framesPerTrial_alldata') % elements of these 2 arrays correspond to each other.
    end
    % it makes a lot of sense to save the above 2 vars for days with a
    % single session too. It will make things much easier! you need to have
    % a framesPerTrial array that corresponds to alldata, and has nan for
    % non-imaged trials. I think you just need to set a nan array same
    % length as the merged alldata (which I guess lacks the last trial),
    % and then fill it with framesPerTrial using trialNumbers as indeces.
end

% clear spikes activity dFOF


%%
%{
load(moreName, 'badROIs01')
%{
% Use below if you want to change what measures % define a bad component
(ie redifine badROIs01)
load(moreName, 'bad_EP_AG_size_tau_tempCorr_hiLight')
badAll = sum(bad_EP_AG_size_tau_tempCorr_hiLight(:,[2:4]),2);
badROIs01 = (badAll ~= 0); % any of the above measure is bad.
%}
goodinds = ~badROIs01; % goodinds = true(size(C,1),1);
%}


%% In alldata, set good quality traces.

% alldataSpikesGood = cellfun(@(x)x(:, goodinds), {alldata.spikes}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
% alldataActivityGood = cellfun(@(x)x(:, goodinds), {alldata.activity}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
alldataDfofGood = cellfun(@(x)x(:, goodinds), {alldata.dFOF}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.












%%

load(imfilename, 'trs2rmv')

postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));

load(postName, 'allResp_HR_LR', 'outcomes', 'stimrate', 'cb')

%{
load(postName, 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
        'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
        'timeReward', 'timeCommitIncorrResp', 'timeStop', ...
        'allResp_HR_LR', 'outcomes', 'stimrate', 'cb')
%}
   

%%
%%%%%% plot average imaging traces aligned on trial events
cprintf('blue', 'Plot average imaging traces and wheel revolution aligned on trial events\n')
evT = {'time1stSideTry'}; %{'1', 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
%     'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
%     'timeReward', 'timeCommitIncorrResp', 'timeStop'};

unitFrame = 0; %1; % if 1, x axis will be in units of frames. If 0, x axis will be in units of time.


%% Set initial vars

%{
outcome2ana = 'all'; % 'all'; 1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
stimrate2ana = 'all'; % 'all'; 'HR'; 'LR';
strength2ana = 'all'; % 'all'; 'easy'; 'medium'; 'hard';    
respSide2ana = 'all'; % 'all'; 'HR'; 'LR';    
thStimStrength = 3; % 2; % threshold of stim strength for defining hard, medium and easy trials.

% remember 1stIncorrTry is not necessarily outcomes==0... depending on how
% the commit lick went.

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

if strcmp(outcome2ana, 'all')
    os = sprintf('%s', outcome2ana);
    outcome2ana = -5:1;    
else
    os = sprintf('%i', outcome2ana);
end

switch stimrate2ana
    case 'HR'
        sr2ana = s > 0;
    case 'LR'
        sr2ana = s < 0;
    otherwise
        sr2ana = true(1, length(outcomes));
end

switch respSide2ana
    case 'HR'
        side2ana = 1;
    case 'LR'
        side2ana = 0;
    otherwise
        side2ana = 'all';
end

if strcmp(side2ana, 'all')
    trs2ana = (ismember(outcomes, outcome2ana)) & str2ana & sr2ana;
else
    trs2ana = (ismember(outcomes, outcome2ana)) & str2ana & sr2ana & (allResp_HR_LR == side2ana);
end
% trs2ana = outcomes==0 & timeLastSideLickToStopT > 1000;
% trs2ana(163:end) = false; % first trials 
% trs2ana(1:162) = false; % last trials

top = sprintf('%s outcomes, %s strengths, %s stimulus: %i trials', os, strength2ana, stimrate2ana, sum(trs2ana));
disp(['Analyzing ', top])

%{
if strcmp(outcome2ana, 'all')    
    trs2ana = str2ana;
    os = sprintf('%s', outcome2ana);
else
    trs2ana = (outcomes==outcome2ana) & str2ana;
    os = sprintf('%i', outcome2ana);
end
%}

%%%
eventsToPlot = 'all'; % 10; % what events to align trials on and plot? % look at evT (below) to find the index of the event you want to plot.
if strcmp(eventsToPlot , 'all')
    ievents = 1:length(evT);
else
    ievents = eventsToPlot;
end
%}

%%
% Set event times (ms) relative to when bcontrol starts sending the scope TTL. event times will be set to NaN for trs2rmv.
%{
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv);
%}
close all

alldatanow = alldata; %alldata(trs2ana);
trs2rmvnow = trs2rmv; %find(ismember(find(trs2ana), trs2rmv));
doplots = 0;


%% Align traces, average them, and plot them.

%{
fn = sprintf('%s - Shades: stand error across trials (neuron-averaged traces of trials)', top);
% fn = sprintf('%s - Shades for rows 1 & 2: standard errors across neural traces (trial-averaged traces of neurons).  Shades for row 3: stand error across trials', top);
f = figure('name', fn, 'position', [680         587        1375         389]);
cnt = 0;
%}
for i = 1 %6 %ievents
    
%     cnt = cnt+1;

    % for licks we get the event times with scopeTTLOrigTime = 0; so they
    % change ... so we need to reload them here!
    load(postName, 'time1stSideTry')%'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
%             'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
%             'timeReward', 'timeCommitIncorrResp', 'timeStop')

    % set DF/F traces (C: temporal component) and wheel traces
    
    disp(['------- ', evT{i}, ' -------'])
    
    eventTime = eval(evT{i});      
    % Take only trials in trs2ana for analysis
%     if length(eventTime)>1
%         eventTime = eventTime(trs2ana);
%     end
    
    if ~iscell(eventTime) && isempty(eventTime), error('No trials!'), end    
    if (iscell(eventTime) && all(cellfun(@(x)all(isnan(x)), eventTime))) || (~iscell(eventTime) && all(isnan(eventTime)))
        warning('Choose outcome2ana and evT wisely! eg. you cannot choose outcome2ana=1 and evT="timeCommitIncorrResp"')
        error('Something wrong; eventTime is all NaNs. Could be due to improper choice of outcome2ana and evT{i}!')
    end
    
    traces = alldataDfofGood; % alldataSpikesGood; %  traces to be aligned.
%     traces = traces(trs2ana);
    
    
    %% Align wheel traces
    
    alignWheel = 1; %nan; % only align the wheel traces
    printsize = 1;
%     traces = [];
    
    [traceEventAlign, timeEventAlign, nvalidtrs, ...
        traceEventAlign_wheelRev, timeEventAlign_wheelRev, nvalidtrs_wheel] ...
        = avetrialAlign_noTrGroup(eventTime, traces, alldatanow, frameLength, trs2rmvnow, alignWheel, printsize, doplots, 1);
    
    
    %% Align lick traces
    
    % Get the time of events in ms relative to the start of each trial in bcontrol
    scopeTTLOrigTime = 0; % above we need event times with scopeTTLOrigTime = 1;
    
    [timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
        time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
        setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime, [], outcomes);
    
    eventTime = eval(evT{i});      
    
    
    %%%%%%%%%%%%%%%%%%
    lickInds = [1, 2,3]; % Licks to analyze % 1: center lick, 2: left lick; 3: right lick
    %     cprintf('blue', 'Plot average lick traces aligned on trial events.\n')    
    
    %%% Set the traces for licks (in time and frame resolution)
    [traces_lick_time, traces_lick_frame] = setLickTraces(alldata, outcomes);

    
    %%% Align    
    traces = traces_lick_time;
    traces = cellfun(@(x)ismember(x, lickInds), traces, 'uniformoutput', 0); % Only extract the licks that you want to analyze (center, left or right).
%     traces = traces(trs2ana);
    
    % it has lots of nans... nPrenPost is better... below
%     [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces, eventTime);

    nPreFrames =[]; nPostFrames = [];
    [traceEventAlign_lick, timeEventAlign_lick, eventI_lick, nPreFrames, nPostFrames] ...
    = triggerAlignTraces_prepost(traces, round(eventTime), nPreFrames, nPostFrames); %, shiftTime, scaleTime, 1); % frames x units x trials        

    cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', 'firstSideTry', nPreFrames, nPostFrames)
    
    
    %% Save wheelRev vars
    
%     check npre and npost for wheelRev... how does it compare to how you get Al traces like firstSideTryAl ... it is like nPre , nPost = []        
    if i==1 %6
        firstSideTryAl_wheelRev.traces = traceEventAlign_wheelRev;
        firstSideTryAl_wheelRev.time = timeEventAlign_wheelRev;
        firstSideTryAl_wheelRev.eventI = nvalidtrs_wheel; 
        firstSideTryAl_wheelRev

        firstSideTryAl_lick.traces = traceEventAlign_lick;
        firstSideTryAl_lick.time = timeEventAlign_lick;
        firstSideTryAl_lick.eventI = eventI_lick; 
        firstSideTryAl_lick
        
        save(postName, '-append', 'firstSideTryAl_wheelRev', 'firstSideTryAl_lick')
    end

    
    
    %% Plots    
    
    if doplots
        
        % Plot wheel revolution
        
        tr00 = traceEventAlign_wheelRev;
        trt00 = timeEventAlign_wheelRev;
        
        figure; hold on
    %     subplot(3,length(ievents),length(ievents)*2+cnt), hold on

        tw = tr00;
    %     tw = tr00(:,:,bb);
        top = nanmean(tw,3); % average across trials
        tosd = nanstd(tw,[],3);
        tosd = tosd / sqrt(size(tw,3)); % plot se

        if unitFrame
            e = find(trt00 >= 0, 1);
            boundedline((1:length(top))-e, top, tosd, 'alpha')
        else
            boundedline(trt00, top, tosd, 'alpha')
        end
        %     plot(top)
        %     plot(trt00, top)

%         xl1 = find(nvalidtrs_wheel >= round(max(nvalidtrs_wheel)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
%         xl2 = find(nvalidtrs_wheel >= round(max(nvalidtrs_wheel)*4/4), 1, 'last');
        xl1 = 1; 
        xl2 = length(trt00); 
        
        if unitFrame
            xlim([xl1 xl2]-e)
        else
            xlim([trt00(xl1)  trt00(xl2)])
        end

        if unitFrame
            plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
            xlabel('Frame')
        else
            plot([frameLength/2  frameLength/2], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
            xlabel('Time (ms)')
        end
        ylim([min(top(xl1:xl2)-tosd(xl1:xl2))  max(top(xl1:xl2)+tosd(xl1:xl2))])
        ylabel('Wheel revolution')       
    
        
        %% Plot lick traces

        figure; hold on; %(f)
    %     subplot(2,length(evT),length(evT)*0+i), hold on

        av = nanmean(traceEventAlign_lick,3); % frames x units. (average across trials).
        top = nanmean(av,2); % average across neurons.
    %     tosd = nanstd(av, [], 2);
    %     tosd = tosd / sqrt(size(av, 2)); % plot se

        e = find(timeEventAlign_lick >= 0, 1);
    %     boundedline((1:length(top))-e, top, tosd, 'alpha')
        plot((1:length(top))-e, top)
        %     plot(top)
        %     plot(timeEventAlign, top)

    %     xl1 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
    %     xl2 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'last');
        xl1 = 1; 
        xl2 = length(timeEventAlign_lick); 

        xlim([xl1 xl2]-e)

        plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')

        %{
        top = nanmean(nanmean(traceEventAlign,3),2); % average across trials and neurons.
        plot(top)
        %     plot(timeEventAlign, top)

        xl1 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
        xl2 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'last');

        %     plot([0 0],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
        e = find(timeEventAlign >= 0, 1);
        plot([e e], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r-.')

        xlim([xl1-100 xl2+100])
        %     xlim([xl1 xl2])
        %     xlim([e-1000 e+1000])
        %     xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
        %}
        if i==1
            xlabel('Time (ms)')
            ylabel('Fraction trials with licks')
        end
        if i>1
            title(evT{i}(5:end))
        else
            title(evT{i})        
        end
        ylabel('Fraction trials with licks')
    
        
    end
    
    
end
    