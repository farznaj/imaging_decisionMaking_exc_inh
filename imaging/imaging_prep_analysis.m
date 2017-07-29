function [alldata, alldataSpikesGood, alldataDfofGood, goodinds, good_excit, good_inhibit, outcomes, allResp, allResp_HR_LR, ...
        trs2rmv, stimdur, stimrate, stimtype, cb, timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, ...
        timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, imfilename, pnevFileName] = ....
   imaging_prep_analysis(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength, save_aligned_traces, savefigs, rmvTrsStimRateChanged);
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
allowCorrectResp = 'change'; % 'change'; 'remove'; 'nothing'; % if 'change': on trials that mouse corrected his choice, go with the original response.
uncommittedResp = 'nothing'; % 'change'; 'remove'; 'nothing'; % what to do on trials that mouse made a response (licked the side port) but did not lick again to commit it.

plot_ave_trGroup = 0; % Plot average traces across all neurons for different trial groups aligned on particular trial events.
plotTrialTraces1by1 = 0; % plot traces per neuron and per trial showing all trial events
furtherAnalyses = 0; % analyses related to choicePref and SVM will be performed.


setNaN_goToneEarlierThanStimOffset = 0; % if 1, set to nan eventTimes of trials that had go tone earlier than stim offset... if 0, only goTone time will be set to nan provided that rmv_timeGoTone_if_stimOffset_aft_goTone = 1

%{
autoTraceQual = 0; %1; % if 1, automatic measure for trace quality will be used.
manualExamineTraceQual = 0; % if 0, traceQuality array needs to be saved.
    saveTraceQual = 0; % it will only take effect if manualExamineTraceQual is 1.
    analyzeQuality = [1 2]; % 1(good) 2(ok-good) 3(ok-bad) 4(bad) % trace qualities that will be analyzed. It will only take effect if manualExamineTraceQual is 1.
orderTraces = 0; % if 1, traces will be ordered based on the measure of quality from high to low quality.
%}

thbeg = 5; % n initial trials to exclude.
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


%% Some plots and prints of info related to behavior and imaging of behavior

%{
* stimulus is played for stimDuration which equal stimDur_diff (or if it is 0, waitDur) + extrastim_dur + stimdur_aftrew .
bur remember if the animal made the choice earlier than the end of stim_duration, then the actual stimulus duration is shorter than this value, bc stim gets stopped when the animal makes a choice.
but if rmv_timeGoTone_if_stimOffset_aft_goTone = 1, you will remove all 
* total waitdur (ie since stim onset, when mouse was allowed to do cent commit) equals waitDuration + postStimDelay
% The following 2 are the same:
figure; plot([alldata.stimDuration])
figure; plot([alldata.stimDur_diff]+[alldata.stimDur_aftRew]+[alldata.extraStimDuration])
%}


% use the following plot to decide on thbeg: number of begining trials that
% you want to exclude.
% stimDur_diff
fhand = figure;
subplot(221), hold on
% title(sprintf('thbeg = %d', thbeg))
title({'Actual stim duration will be shorter if','the final state happens earlier than','alldata.stimDuration'})
plot([all_data.extraStimDuration])
plot([all_data.stimDur_aftRew])
plot([all_data.stimDur_diff])
plot([all_data.waitDuration]+[all_data.postStimDelay])
plot([all_data.stimDuration])
% plot([thbeg thbeg],[-1 12], 'k')
legend('extraStimDur', 'stimDur\_aftRew', 'stimDur\_diff', 'waitDur', 'stimDuration')
set(gcf, 'name', 'stimDuration= stimDur_diff (or if it is 0, waitDur) + extrastim_dur + stimdur_aftrew .')
m = max([all_data.stimDuration]);
ylim([-1 m+1])
set(gca,'tickdir','out')


% u = unique({all_data.rewardStage}); disp(u);
rs = {all_data.rewardStage}; 
ave_rewardStage_allowCorr_chooseSide = [nanmean(strcmp(rs, 'Allow correction')), nanmean(strcmp(rs, 'Choose side'))];
fprintf('Fraction of trials (rewardStage): allowCorr= %.3f. chooseSide= %.3f (GUI-set, not necessarily used by the animal.)\n', ave_rewardStage_allowCorr_chooseSide)
% rsi = zeros(1, length(rs)); 
% rsi(strcmp(rs, 'Choose side')) = 1; 
% figure; plot(rsi); ylim([-.1 1.1])
% xlabel('Trials'), ylabel('chooseSide'), box off, set(gca,'tickdir','out')


%%
%%%%%%%%%%%%%%%%%%%%%%%%% Load neural data, merge them into all_data, and assess trace quality %%%%%%%%%%%%%%%%%%%%%%%

%% Load vars related to manual method

load(pnevFileName, 'activity_man_eftMask_ch2')

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

figure(fhand), subplot(223); hold on;
plot(outputsDFT{1}(:,3:4))
yax = get(gca,'ylim');
plot(badFrames{1}*range(yax))
plot(pmtOffFrames{1}*range(yax))

xlabel('Frame')
ylabel('Pixel shift')
set(gcf, 'name', ['+X : brain moved left(lateral) ;+Y : brain moved up(posterior bc of 2p I think)'])
legend('Y (row) shift', 'X (column) shift', 'badFrames', 'pmtOffFrames')
pof = cellfun(@sum, pmtOffFrames);
bf = cellfun(@sum, badFrames);
t1 = sprintf('%d pmtOffFrames', pof(1));
t2 =  sprintf('%d badFrames', bf(1));
title({t1,t2})

pmtOffFrames0 = pmtOffFrames;
badFrames0 = badFrames;


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
%{
% num frames per session:
cs_sess = 0; 
for imdf = 1:length(mdfFileNumber)    
    cs_sess = [cs_sess, sum(nFrsMov{imdf})];
end
cs_sess = cumsum(cs_sess);
%}
% end

%{
% Find the tif movie that contains a frame:
% (going from a frame on the entire movie to that same frame on the tif movie that contains the frame)

% find the index of a frame on the tif movie that the frame belongs to
% (by using the frame index on the entire movie) ... ie match frame
% indeces between entire movie (iall) and the tif movie that contains the
% frame (imov):
iall = 34998; % frame index on the entire movie (eg on S)
itif = find((cs_frmovs - iall)>=0, 1)-1 % tif movie containing frame iall
imov = iall - cs_frmovs(itif) % frame index on the movie itif 

% Find the trial that contains frame iall
itr = find((cs_frtrs - iall)>=0, 1)-1; % trial that contains frame iall 
% Find the frame index of a trial (its onset) on movie itif
cs_frtrs(itr) - cs_frmovs(itif) % frame of trial itr on movie itif (ie frame index of trial itr relative to
% the beginning of movie itif)
%}

    
%% Evaluate C,f,manual activity, also tau, sn as well as some params related to A

if ~exist(fullfile(pd, 'figs'), 'dir')
    mkdir(fullfile(pd, 'figs')) % save the following 3 figures in a folder named "figs"
end

if evaluateEftyOuts
    
    load(imfilename, 'imHeight', 'imWidth', 'medImage')    
    
    % Plot average of spatial components.
    figure; imagesc(reshape(mean(A,2), imHeight, imWidth))  %     figure; imagesc(reshape(mean(P.psdx,2), imHeight, imWidth))

    % Plot COMs
    im = normImage(medImage{2});
    load(moreName, 'CC')
    COMs = fastCOMsA(A, [imHeight, imWidth]);
    
    figure;
    imagesc(im); hold on; % imagesc(log(im));   %     colormap gray     
    for rr = 1:length(CC) % find(~badROIs01)'                 
        plot(COMs(rr,2), COMs(rr,1), 'r.')
    end   
%     im = im - min(im(:)); softImMax = quantile(im(:), 0.995); im = im / softImMax; im(im > 1) = 1; % matt        
    %{
    contour_threshold = .95;
    plotCOMs = 1;
    [CC, ~, COMs] = setCC_cleanCC_plotCC_setMask(A, imHeight, imWidth, contour_threshold, im, plotCOMs);
    %}


    %% plot C, f, manual activity
    
    load(imfilename, 'cs_frtrs')    
    
    plotEftyVarsMean(mouse, imagingFolder, mdfFileNumber, {C, S, f, activity_man_eftMask_ch2, cs_frtrs})
    
    if savefigs
        savefig(fullfile(pd, 'figs','caTraces_aveAllNeurons'))  
    end
    
    
    %% shift and scale C, man, etc to compare them...        
    
    load(pnevFileName, 'activity_man_eftMask_ch1')
    
    figure('name', 'Green lines: trial beginnings. Solid black lines: session beginnings. Dashed black lines: tif movie beginnings.'); 
    subplot(311), hold on
    h = plot([cs_frtrs; cs_frtrs], [-.3; .5], 'g'); % mark trial beginnings
    if exist('nFrsSess', 'var'), 
        h0 = plot([cumsum([0, nFrsSess]); cumsum([0, nFrsSess])], [-.5; 1], 'k'); % mark session beginnings
        h00 = plot([cs_frmovs; cs_frmovs], [-.5; 1], 'k:'); % mark tif movie beginnings
        set([h0; h00], 'handlevisibility', 'off'); 
    end 
    set([h], 'handlevisibility', 'off')
    plot(shiftScaleY(f), 'color', [255,215,0]/255)
    plot(shiftScaleY(mean(activity_man_eftMask_ch2,2)), 'b')
    plot(shiftScaleY(mean(C)), 'k')
    plot(shiftScaleY(mean(S)), 'm')
    xlim([1 size(C,2)]) % xlim([1 1500])
    legend('f','manAct', 'C', 'S')
    title('Average of all neurons')
    a1 = gca;
    
    
    subplot(312), hold on
    h = plot([cs_frtrs; cs_frtrs], [min(mean(C_df)); max(mean(C_df))], 'g'); % mark trial beginings
    if exist('nFrsSess', 'var'), h0 = plot([cumsum([0, nFrsSess]); cumsum([0, nFrsSess])], [-.5; 1], 'k'); set([h0], 'handlevisibility', 'off'); end % mark session beginings
    set([h], 'handlevisibility', 'off')
    plot(mean(C_df), 'k')
    xlim([1 size(C,2)]) % xlim([1 1500])
    legend('C\_df')
    a2 = gca;
    
    
    subplot(313), hold on
    plot(mean(activity_man_eftMask_ch1,2))
    a3 = gca;
    legend('Ch1')
    
    linkaxes([a1, a2, a3], 'x')    
    
    if savefigs
        savefig(fullfile(pd, 'figs','caTraces_aveAllNeurons_sup'))  
    end
    
    
    %% Assess tau and noise for each neuron
    % load(pnevFileName, 'P')
    figure;
    subplot(321), plot(tau(:,2)), xlabel('Neuron'), ylabel('Tau (ms)'), legend('decay'), xlim([0 size(A,2)+1]) %, legend('rise','decay')
    subplot(323), plot(cell2mat(P.neuron_sn)), xlabel('Neuron'), ylabel('neuron\_sn'), xlim([0 size(A,2)+1])
    subplot(325), histogram(tau(:,2)), xlabel('Decay tau (ms)'), ylabel('# Neurons')


    % Assess a number of parameters related to A (spatial component)
    % load(pnevFileName, 'A')
%     figure;
    subplot(322), plot(sum(A~=0,1)), ylabel('Number of pixels in A'), xlim([0 size(A,2)+1])
    subplot(324), plot(nanmean(A,1)), ylabel('Mean A'), xlim([0 size(A,2)+1])
    subplot(326), plot(max(A,[],1)), ylabel('Max A'), xlim([0 size(A,2)+1])
    xlabel('Neuron')
    
end


%% Evaluate A and C of Efty's algorithm by plotting the traces 1 by 1.

if plotEftyAC1by1
    
%     load(moreName, 'CC')    
    load(imfilename, 'imHeight', 'imWidth', 'medImage', 'cs_frtrs')
    im = medImage{2};
    im = im - min(im(:)); softImMax = quantile(im(:), 0.995); im = im / softImMax; im(im > 1) = 1; % matt
    
%     C_df0 = konnerthDeltaFOverF(C', pmtOffFrames{gcampCh}, smoothPts, minPts);
%     C_df = C_df0';
    
    inds2plot = randperm(size(C,1)); % 1:size(C,1); % randperm(size(C,1)) % excl(randperm(length(excl)))'; % size(C,1):-1:1;  
    
    % plot C and act_man
    plotEftManTracesROIs(C, S, activity_man_eftMask_ch2', A, [], CC, [], 1:size(C_df,1), im, C, inds2plot, 0, 0, medImage{1}, cs_frtrs);
     
    % below is fine too, if you want to plot df/f 
    %{
    if ~exist('dFOF_man','var') % use this if you don't have manual activity
        plotEftManTracesROIs(C_df, S, [], A, [], CC, [], [], im, C, inds2plot, 0, 0, medImage{1}, cs_frtrs);
    else % use this if you wan to compare with manual activity:
        plotEftManTracesROIs(C_df, S, dFOF_man', A, [], CC, [], 1:size(C_df,1), im, C, inds2plot, 0, 0, medImage{1}, cs_frtrs);       
%         plotEftManTracesROIs(C_df, S, activity_man_eftMask_ch2', A, [], CC, [], 1:size(C_df,1), im, C, inds2plot, 0, 0, medImage{1}, cs_frtrs);  % use this if for the superimposed image you want to compare C with manActiv (instead of their df versions)
        % traceQualManual = plotEftManTracesROIs(C_df, S_df, dFOF, A2, mask_eft, CC, CC_rois, eftMatchIdx_mask, im, C, inds2plot, manualTraceQual, plothists, im2)
    end
    %}
    
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
    
    multi_sess_set_vars
end

clear spikes activity dFOF


%% Plot mscan lag (mscan lag should be ok bc you control for it when you compute all data.frameTimes? 
% badAlign is different: if the end of sampleOff does not align with the beginning of codeTime, something is wrong, so badAlign must be excluded.

save(imfilename, '-append', 'mscanLag', 'trs2rmv')

figure; plot(mscanLag), xlabel('Trial'); ylabel('MScan lag (ms)')
if savefigs
    savefig(fullfile(pd, 'figs','MScanLag')) 
end

fl = find(abs(mscanLag) > 32);
if ~isempty(fl)
    fprintf('%d trials have mscanLag > 32ms\n', length(fl))
    fll = fl(~ismember(fl, trs2rmv));
    if ~isempty(fll)
        warning('\t%d trial(s) with long mscanLag are not in trs2rmv', length(fll))
        fprintf('mscanLag = %d ms\t\n', mscanLag(fll))
    else
        fprintf('\tall of them are in trs2rmv. Good!\n')
    end
end


%% Set outcome and response side for each trial, taking into account allowcorrection and uncommitted responses.

% Set some params related to behavior % behavior_info
%{
allowCorrectResp = 'change'; % 'change'; 'remove'; 'nothing'; % if 'change': on trials that mouse corrected his choice, go with the original response.
uncommittedResp = 'nothing'; % 'change'; 'remove'; 'nothing'; % what to do on trials that mouse made a response (licked the side port) but did not lick again to commit it.
%}
allowCorrectOutcomeChange = 1;
verbose = 1;
[outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp, allowCorrectOutcomeChange, verbose); % 1 for HR choice, 0 for LR choice.

% set trs2rmv to nan
outcomes(trs2rmv) = NaN;
allResp(trs2rmv) = NaN;
allResp_HR_LR(trs2rmv) = NaN;
% stimrate(trs2rmv) = NaN;

%%%%% Save outcomes and allResp to a file named post_pnev... 
postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
if ~exist(postName, 'file')
    warning('creating postFile and saving vars in it')
    save(postName, 'outcomes', 'allResp_HR_LR', 'stimrate', 'rmvTrsStimRateChanged')
end

% save('151102_001.mat', '-append', 'trs2rmv')  % Do this!

% imaging.trs2rmv = trs2rmv;
% save(imfilename, '-append', 'imaging')
    
figure(fhand)
subplot(222)
plot(outcomes), xlabel('Trials'), ylabel('Outcome'), %  1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
subplot(224)
plot(allResp_HR_LR), ylim([-.5 1.5]), xlabel('Trials'), ylabel('Response (HR:1 , LR: 0)')
if savefigs
    savefig(fullfile(pd, 'figs','behav_motCorr_sum')) 
end

a = matfile(postName);
if ~isprop(a, 'alldata_frameTimes')
    alldata_frameTimes = {alldata.frameTimes};
    save(postName, '-append', 'alldata_frameTimes')
end


%% Set event times (ms) relative to when bcontrol starts sending the scope TTL. event times will be set to NaN for trs2rmv.

%{
% best is to set the 2 vars below to 0 so u get times of events for all trials; later decide which ones to set to nan.
rmv_timeGoTone_if_stimOffset_aft_goTone = 0; % if 1, trials with stimOffset after goTone will be removed from timeGoTone (ie any analyses that aligns trials on the go tone)
rmv_time1stSide_if_stimOffset_aft_1stSide = 0; % if 1, trials with stimOffset after 1stSideTry will be removed from time1stSideTry (ie any analyses that aligns trials on the 1stSideTry)
setNaN_goToneEarlierThanStimOffset = 0; % if 1, set to nan eventTimes of trials that had go tone earlier than stim offset... if 0, only goTone time will be set to nan.

%}
scopeTTLOrigTime = 1;
stimAftGoToneParams = {rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, setNaN_goToneEarlierThanStimOffset};
% stimAftGoToneParams = []; % {0,0,0};
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, timeStimOnsetAll, timeSingleStimOffset] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime, stimAftGoToneParams, outcomes);

% below are problematic trials in which go tone happened earlier than
% stimulus offset... you need to take care of them for your analyses!
% trsGoToneEarlierThanStimOffset = find(timeCommitCL_CR_Gotone < timeStimOffset)';

% alldata_frameTimes = {alldata.frameTimes};
% save(imfilename, '-append', 'alldata_frameTimes', 'timeStop')



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Take care of neural traces %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Assess trace quality of each neuron

load(moreName, 'badROIs01')
%{
% Use below if you want to change what measures % define a bad component
(ie redifine badROIs01)
load(moreName, 'bad_EP_AG_size_tau_tempCorr_hiLight')
badAll = sum(bad_EP_AG_size_tau_tempCorr_hiLight(:,[2:4]),2);
badROIs01 = (badAll ~= 0); % any of the above measure is bad.
%}
goodinds = ~badROIs01; % goodinds = true(size(C,1),1);


%{
%{
mouse = 'fni17';
imagingFolder = '151102'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1,2];  % 3; %1; % or tif major
%}
fixed_th_srt_val = 1; % if fixed 4150 will be used as the threshold on srt_val, if not, we will find the srt_val threshold by employing Andrea's measure
savebadROIs01 = 0; % if 1, badROIs01 will be appended to more_pnevFile
exclude_badHighlightCorr = 1;
evalBadRes = 0; % plot figures to evaluate the results

th_AG = -20; % you can change it to -30 to exclude more of the poor quality ROIs.
th_srt_val = 4150;
th_smallROI = 15;
th_shortDecayTau = 200;
th_badTempCorr = .4;
th_badHighlightCorr = .5;

[badROIs01, bad_EP_AG_size_tau_tempCorr_hiLight] = findBadROIs(mouse, imagingFolder, mdfFileNumber, fixed_th_srt_val, savebadROIs01, exclude_badHighlightCorr,evalBadRes, th_AG, th_srt_val, th_smallROI, th_shortDecayTau, th_badTempCorr, th_badHighlightCorr);
%}

% If you don't exclude badHighlightCorr, still most of the neurons will
% have good trace quality, but they are mostly fragmented parts of ROIs or
% neuropils. Also remember in most cases of fragmented ROIs, a more
% complete ROI already exists that is not a badHighlightCorr.


% Old method
%{
if manualExamineTraceQual
    %     assessCaTraceQaulity % another script to look at traces 1 by 1.
    
    inds2plot = randperm(size(activity,2));
    traceQuality = plotEftManTracesROIs(dFOF', spikes', [], [], [], [], [], [], [], activity', inds2plot, 1, 0, []);
    
    if saveTraceQual
        save(imfilename, 'traceQuality', '-append')
    end
    goodNeurons = ismember(traceQuality, analyzeQuality);
end


%%%%% Automatic assessment of trace quality

% Compute measures of trace quality
if autoTraceQual
    [avePks2sdS, aveProm2sdS, measQual] = traceQualMeasure(dFOF', spikes');
    
    % badQual = find(avePks2sdS<3 | aveProm2sdS<1);
    badQual = find(measQual<0);
    
    goodNeurons = true(1,length(avePks2sdS));
    goodNeurons(badQual) = false;
    
    % look at C of neurons identified as badQual.
    %     plotCaTracesPerNeuron({dFOF(:, badQual)}, [], [], 0, 0, 0, 0)
    
    figure;
    for i = badQual, % 1:size(dFOF)
        plot(dFOF(:,i))
        %         plot(spikes(:,i))
        
        set(gcf,'name',num2str(i))
        title(sprintf('pks %.2f   prom %.2f   prod %.2f   meas %.2f', avePks2sdS(i), aveProm2sdS(i), ...
            avePks2sdS(i) * aveProm2sdS(i), measQual(i)))
        
        pause
    end
    
    if orderTraces
        % Sort traces
        [sPks, iPks] = sort(avePks2sdS);
        [sProm, iProm] = sort(aveProm2sdS);
        [sPksProm, iPksProm] = sort(avePks2sdS .* aveProm2sdS);
        [sMeasQual, iMeasQual] = sort(measQual);
        
        % order based on automatic measure
        goodnessOrder = iMeasQual(sMeasQual>=0); % ascending order of neurons based on trace quality
        goodnessOrder = goodnessOrder(end:-1:1); % descending order
    end
end
% fprintf('N good-quality and all neurons: %d, %d. Ratio: %.2f\n', [sum(goodNeurons), size(activity,2), sum(goodNeurons)/ size(activity,2)]); % number and fraction of good neurons vs number of all neurons.


% set goodinds: an array of length of all neurons, with 1s indicating good and 0s bad neurons.
if ~any([autoTraceQual, manualExamineTraceQual])  % if no trace quality examination (auto or manual), then include all neurons.
    goodinds = true(1, size(dFOF,2)); %    goodnessOrder = 1:size(dFOF,2);    goodNeurons = 1:size(dFOF,2);
elseif orderTraces % descending based on the measure of quality
    error('the code below needs work. goodinds is supposed to be logical not tr numbers.')
    goodinds = goodnessOrder;
else % stick to the original order
    goodinds = goodNeurons;
end

fprintf('N good-quality and all neurons: %d, %d. Ratio: %.2f\n', [sum(goodinds), size(activity,2), sum(goodinds)/ size(activity,2)]); % number and fraction of good neurons vs number of all neurons.
%}


%% In alldata, set good quality traces.

alldataSpikesGood = cellfun(@(x)x(:, goodinds), {alldata.spikes}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
alldataActivityGood = cellfun(@(x)x(:, goodinds), {alldata.activity}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
alldataDfofGood = cellfun(@(x)x(:, goodinds), {alldata.dFOF}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.


% The following is all good, but commented to save memory. uncomment if you need the vars.
%{
activityGood = activity(:, goodinds); % frames x units % remember activity and dFOF may have more frames that alldataDfofGood_mat bc alldataDfofGood_mat does not include the frames of the trial during which mscan was stopped but activity includes those frames.
dfofGood = dFOF(:, goodinds); % frames x units
spikesGood = spikes(:, goodinds); % frames x units

alldataDfofGood_mat = cell2mat(alldataDfofGood'); % frames x units;  % same as dfofGood except for some frames at the end that it may miss due to stopping mscan at the middle of a trial.

if compareManual
    eftMatchIdx_mask_good = matchedROI_idx(goodinds);
end
%}

% alldataDfof = {alldata.dFOF};
% alldataDfof_mat = cell2mat(alldataDfof'); % frames x units


% numUnits = size(alldata(1).dFOF,2); % all units, regardless of the quality.
% numGoodUnits = size(dfofGood,2);
% numTrials = length(alldata);


%% Set inhibit and excit traces.

% alldataDfofGoodInh, alldataSpikesGoodInh and alldataDfofGoodExc, alldataSpikesGoodExc.
% size: 1 x number of trials, each element includes the traces of all inh (exc)
% neurons.


if setInhibitExcit    
    
    load(moreName, 'inhibitRois')
%     inhibit_excit_prep % preps vars and calls inhibitROIselection
    
    % Set good_inhibit and good_excit neurons (ie in good quality neurons which ones are inhibit and which ones are excit).
    
    good_inhibit = inhibitRois==1;
    good_excit = inhibitRois==0;
    
    % use below if you find inhibitory neurons on all (good and bad) ROIs.
    %{
    % goodinds: an array of length of all neurons, with 1s indicating good and 0s bad neurons.
    good_inhibit = inhibitRois(goodinds) == 1; % an array of length of good neurons, with 1s for inhibit. and 0s for excit. neurons and nans for unsure neurons.
    good_excit = inhibitRois(goodinds) == 0; % an array of length of good neurons, with 1s for excit. and 0s for inhibit neurons and nans for unsure neurons.
    
    % you can use the codes below if you want to be safe:
    %     good_inhibit = inhibitRois(goodinds & roi2surr_sig >= 1.3) == 1;
    %     good_excit = inhibitRois(goodinds & roi2surr_sig <= 1.1) == 0;
    
    if nanmean(goodinds) ~= 1
        fprintf('Fract inhibit in all, good & bad Ns = %.3f  %.3f  %.3f\n', [...
            nanmean(inhibitRois==1)
            nanmean(inhibitRois(goodinds)==1)
            nanmean(inhibitRois(~goodinds)==1)])
        % it seems good neurons include more excit (than inhibit) neurons
        % as if for some reason tdtomato neruons have low quality on the
        % green channel.
    end
    %}
    
    cprintf('blue', 'Fract inhibit %.3f, excit %.3f in good quality neurons\n', [nanmean(good_inhibit), nanmean(good_excit)])
    
    
    %% Set traces for good inhibit and excit neurons.
    
    % The following is all good, but commented to save memory. uncomment if you need the vars.
    %{
    alldataDfofGoodInh = cellfun(@(x)x(:, good_inhibit==1), alldataDfofGood, 'uniformoutput', 0); % 1 x number of trials
    alldataSpikesGoodInh = cellfun(@(x)x(:, good_inhibit==1), alldataSpikesGood, 'uniformoutput', 0); % 1 x number of trials
    
    alldataDfofGoodExc = cellfun(@(x)x(:, good_excit==1), alldataDfofGood, 'uniformoutput', 0); % 1 x number of trials
    alldataSpikesGoodExc = cellfun(@(x)x(:, good_excit==1), alldataSpikesGood, 'uniformoutput', 0); % 1 x number of trials
    %}
    
    % For the matrices below just do (:, good_inhibit) and (:, good_excit)
    % to get their corresponding traces for inhibit and excit neurons :
    % activityGood
    % dfofGood
    % spikesGood
    % alldataDfofGood_mat
    
else
    good_excit = [];
    good_inhibit = [];
end



%%
%%%%%%%%%%%%%%%%%%%%%%%%% Do some plotting of neural traces %%%%%%%%%%%%%%%%%%%%%%%

%% Look at traces per neuron and per trial and show events during a trial as well. Traces start at the scan start (so not aligned on any particular event.)

if plotTrialTraces1by1
    
    % it makes sense to look at all trials and not just ~trs2rmv so do
    % trs2rmv = unique([trs_problemAlign, trs_badMotion_pmtOff]);, and again run setEvent
    plotTrs1by1 = 0; %1;
    interactZoom = 0;
    markQuantPeaks = 1; % 1;
    showitiFrNums = 1;
    allEventTimes = {timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, timeStop, centerLicks, leftLicks, rightLicks};
    %     [~, ~, stimrate] = setStimRateType(alldata);
    
    % Remember the first input will be plotted!
    % If you want to plot certain neurons, specify that in dfofGood, eg: dFOF(:, badQual)
    plotCaTracesPerNeuron({dfofGood}, alldataDfofGood, alldataSpikesGood, interactZoom, plotTrs1by1, markQuantPeaks, showitiFrNums, {framesPerTrial, alldata, spikesGood, [], [], allEventTimes, stimrate})
    
    % plotCaTracesPerNeuron([{C_df'}, {C_mcmc_df'}], interactZoom, plotTrs1by1, markQuantPeaks, alldataDfofGood, alldataSpikesGood, {framesPerTrial, alldata, S_mcmc', dFOF_man, matchedROI_idx, allEventTimes, stimrate})
    % plot badQual neurons.co
    %     plotCaTracesPerNeuron({dFOF(:, badQual)}, [], [], 0, 0, 0, 0)
    
    
    %% plot all neuron traces per trial
    
    plotCaTracesPerTrial
    
    
end


%% Plot average traces across all neurons and all trials aligned on particular trial events.

if plot_ave_noTrGroup    

    outcome2ana = 'all'; % 'all'; 1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
    stimrate2ana = 'all'; % 'all'; 'HR'; 'LR';
    strength2ana = 'all'; % 'all'; 'easy'; 'medium'; 'hard';    
    respSide2ana = 'all'; % 'all'; 'HR'; 'LR';    
  
    
    %%%%%% plot average imaging traces aligned on trial events
    cprintf('blue', 'Plot average imaging traces aligned on trial events\n')
    evT = {'1', 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
        'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
        'timeReward', 'timeCommitIncorrResp', 'timeStop'};
    
    avetrialAlign_plotAve_noTrGroup(evT, outcome2ana, stimrate2ana, strength2ana, respSide2ana, trs2rmv, allResp_HR_LR, outcomes, stimrate, cb, alldata, alldataDfofGood, alldataSpikesGood, frameLength, timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, timeStop)
    if savefigs
        savefig(fullfile(pd, 'figs','caTraces_trEventAl'))    
    end
    
    %%%%%% plot average lick traces aligned on trial events.
    cprintf('blue', 'Plot average lick traces aligned on trial events.\n')
    lickInds = [1, 2,3]; % Licks to analyze % 1: center lick, 2: left lick; 3: right lick
    
    lickAlign(lickInds, evT, outcome2ana, stimrate2ana, strength2ana, trs2rmv, outcomes, stimrate, cb, alldata, frameLength)    
    if savefigs
        savefig(fullfile(pd, 'figs','lickTraces_trEventAl'))
    end
    
    
    
    %%%%%% plot average imaging traces aligned on licks. THIS TAKES A LOT OF TIME for center licks
    cprintf('blue', 'Plot average imaging traces aligned on licks\n')
%     evT = {'centerLicks', 'leftLicks', 'rightLicks'}; % times are relative to scopeTTL onset, hence negative values are licks that happened before that (during iti states).
    evT = {'leftLicks', 'rightLicks'}; % remove center licks so it takes less time
    nPreFrames = 5;
    nPostFrames = 20;    
    excludeLicksPrePost = 'none'; % 'none'; 'pre'; 'post'; 'both';
    
    avetrialAlign_plotAve_noTrGroup_licks(evT, outcome2ana, stimrate2ana, strength2ana, outcomes, stimrate, cb, alldata, alldataDfofGood, alldataSpikesGood, frameLength, nPreFrames, nPostFrames, centerLicks, leftLicks, rightLicks, excludeLicksPrePost)
    if savefigs
        savefig(fullfile(pd, 'figs','caTraces_lickAl'))    
    end
    
end


%% Plot average traces across all neurons for different trial groups aligned on particular trial events.

if plot_ave_trGroup
    fprintf('Remember: in each subplot (ie each neuron) different trials are contributing to different segments (alignments) of the trace!\n')
    avetrialAlign_plotAve_trGroup
end




%% Set and save aligned traces (will be used for SVM, etc)

% what traces you want to align and plot: (normally you save alldataSpikesGood)

tracesAll = {alldataSpikesGood, alldataActivityGood, alldataDfofGood}; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
traceTypeAll = {'_S_', '_C_', '_dfof_'};

% traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
% traces = alldataDfofGood;
% traces = alldataActivityGood;

varsAlreadySet = 1;
loadPostNameVars = 0; % if 1, aligned traces will be loaded from postName, but we just set them in set_aligned_traces

a = matfile(postName);
if isprop(a, 'firstSideTryAl')==0
    saveAlVars = 1;
else
    saveAlVars = 0;
end

for itrac = 1:3
       
    traceType = traceTypeAll{itrac};
    traces = tracesAll{itrac};

    fprintf('Making plots for the %s traces...\n', traceType(2:end-1))
    
    set_aligned_traces
    
    if itrac~=1
        close(fannoy) 
    end
%     load(postName, 'outcomes') % outcomes based on animal's 1st choice. we need to load them again bc they are changed in set_aligned_traces
    % on July 3, 2017 you made popClassifier_trialHistory a function.
    % Before that outcomes and allResp_HR_LR were getting changed there,
    % but above you only loaded outcomes... so plots below were not
    % quite correct if there was allowCorr.
    
    if save_aligned_traces & itrac==1 & saveAlVars==1 % only save it for S
        save(postName, '-append', 'trialHistory', 'firstSideTryAl', 'firstSideTryAl_COM', 'goToneAl', 'goToneAl_noStimAft', 'rewardAl', 'commitIncorrAl', 'initToneAl', 'stimAl_allTrs', 'stimAl_noEarlyDec', 'stimOffAl')
    end

    
    %% Plot average ca traces (across all neurons) for HR and LR trials, aligned on different trial events. (at the begining of the script you set some vars... check them if you want to customize them!).

    plotAlTracesHrLr
    
    
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%% Start some analyses: alignement, choice preference, SVM %%%%%%%%%%%%%%%%%%%%%%%

if furtherAnalyses
    %% Align traces on particular trial events
    
    % remember traces_al_sm has nan for trs2rmv as well as trs in alignedEvent that are nan.
    
    traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
    alignedEvent = 'stimOn'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'
    dofilter = false; true; % false;
    % set nPre and nPost to nan if you want to go with the numbers that are based on eventBef and eventAft.
    % set to [] to include all frames before and after the alignedEvent.
    nPreFrames = []; % nan;
    nPostFrames = []; % nan;
    
    traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.
    
    [traces_al_sm, time_aligned_stimOn, eventI_stimOn] = alignTraces_prePost_filt...
        (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
        timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);
    
    % set to nan those trials in outcomes and allRes that are nan in traces_al_sm
    a = find(sum(sum(~isnan(traces_al_sm),1),3), 1);
    allTrs2rmv = find(squeeze(sum(isnan(traces_al_sm(:,a,:)))));
    outcomes(allTrs2rmv) = NaN;
    allResp(allTrs2rmv) = NaN;
    allResp_HR_LR(allTrs2rmv) = NaN;
    
    
    %% Compute and plot choice preference, 2*(auc-0.5), for each neuron at each frame.
    
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
    
    %%% Compute choicePref for the average of frames after the stim.
    % traces_al_sm_aveFr = nanmean(traces_al_sm(eventI_stimOn:end,:,:), 1);
    % choicePref_all = choicePref_ROC(traces_al_sm_aveFr, ipsiTrs, contraTrs, makeplots, eventI_stimOn, useEqualNumTrs);
    
    
    %% SVM    
    
    popClassifier
    
    
end


% diary off

%%
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