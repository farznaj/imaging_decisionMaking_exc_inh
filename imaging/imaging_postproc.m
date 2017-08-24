% Right after you are done with preproc on the cluster, run the following scripts:
% - plotMotionCorrOuts
% - plotEftyVarsMean (if needed follow by setPmtOffFrames to set pmtOffFrames and by findTrsWithMissingFrames to set frame-dropped trials. In this latter case you will need to rerun CNMF!): for a quick evaluation of the traces and spotting any potential frame drops, etc
% - eval_comp_main on python (to save outputs of Andrea's evaluation of components in a mat file named more_pnevFile)
% - set_mask_CC
% - findBadROIs
% - inhibit_excit_prep
% - imaging_prep_analysis (calls set_aligned_traces... you will need its outputs)
%{

clear; close all
mouse = 'fni17';
imagingFolder = '150813';
mdfFileNumber = [1];      % or tif major

setFrdrops = 1; % set to 1 write after cnmf, so if needed you reran cnmf
bad_mask_inh = 1; % if 1, badROIs, mask, and inh/exc will be set (ie almost all except for imaging_prep_analysis)
doPost = 1; % if 1 imaging_prep_analysis will be run.

%}


%% Set vars

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
% [pd, pnev_n] = fileparts(pnevFileName);
[md,date_major] = fileparts(imfilename);
cd(md) % r = repmat('%03d-', 1, length(mdfFileNumber)); r(end) = []; date_major = sprintf(['%s_', r], imagingFolder, mdfFileNumber);
%{
[~, pnev_n] = fileparts(pnevFileName);
postName = fullfile(md, sprintf('post_%s.mat', pnev_n));
%}
%{
% rename the old postName file ... remove this later
a = dir('post_*');
for ia = 1:length(a)
    movefile(a(ia).name, ['0_',a(ia).name])
    % delete(a.name)
end
%}


%% Assess motion correction. Also see how normalizing the movie worked (in order to make pixel intensities uniform before running CNMF).

if setFrdrops
    
    showMov = 1; % set to 1 to see MC rep movies.
    plotMotCorr_normImg(mouse, imagingFolder, mdfFileNumber, showMov)


    %% plotEftyVarsMean

    % close all
    cprintf('blue', 'If needed follow this script by codes setPmtOffFrames to set pmtOffFrames\nand by findTrsWithMissingFrames to set frame-dropped trials. In this\nlatter case you will need to rerun CNMF (bc frames were dropped and\nidentification of trials was wrong)!\n')

    doPause = 1;
    plotEftyVarsMean(mouse, imagingFolder, mdfFileNumber, 1, doPause)

    % setPmtOffFrames % set pmtOffFrames
    % findTrsWithMissingFrames % set frame-dropped trials
end


%% Run eval_comp_main from python (to save outputs of Andrea's evaluation of components in a mat file named more_pnevFile)

a = dir('more_*');
if bad_mask_inh && isempty(a)
    
    %%% Start a diary file
    nowStr = datestr(now, 'yymmdd-HHMMSS');
    dn = ['diary_BMI_',date_major, '_',nowStr]; % BMI: bad mask inh :)
    diary(dn)
    
    load(pnevFileName, 'C', 'YrA') % # C and YrA are provided as inputs to eval_comp_main. This helps to call this function from matlab. Because importing h5py causes matlab to crash when python is called from matlab.

    %%%%%%%
    cd ~/Documents/trial_history/imaging
    %{
    if count(py.sys.path,'') == 0
        insert(py.sys.path,int32(0),'');
    end
    py.importlib.import_module('setImagingAnalysisNamesP')

    % py.setImagingAnalysisNamesP.setImagingAnalysisNamesP(mouse, imagingFolder, num2cell(mdfFileNumber))
    % py.eval_comp_main.eval_comp_main(mouse, imagingFolder, num2cell(mdfFileNumber))
    %}

    % for R2017 I had to exclude:
    %{
    % The following need be run if they are modified!
    % mod should be the the .py file (not the .pyc file!! if so delete .pyc file and rerun below)
    mod = py.importlib.import_module('setImagingAnalysisNamesP');
    py.reload(mod)

    mod = py.importlib.import_module('eval_comp_main');
    py.reload(mod)
    %}

    py.eval_comp_main.eval_comp_main(mouse, imagingFolder, num2cell(mdfFileNumber), C(:)', YrA(:)', size(C))
    % figure; subplot(211), plot(fitness); subplot(212), plot(idx_components)

    cd(fileparts(imfilename))
%     clearvars -except mouse imagingFolder mdfFileNumber days


    %% set_mask_CC

    set_mask_CC(mouse, imagingFolder, mdfFileNumber)


    %% findBadROIs

    savebadROIs01 = 1; % if 1, badROIs01 will be appended to more_pnevFile
    evalBadRes = 0; %1; % plot figures to evaluate the results

    if ismember(mouse, {'fni16', 'fni17', 'fni19'})
        fixed_th_srt_val = 1; % it was 1 for ni16,fni17; changed to 0 for fn18. % if fixed 4150 will be used as the threshold on srt_val, if not, we will find the srt_val threshold by employing Andrea's measure
    elseif strcmp(mouse,'fni18')
        fixed_th_srt_val = 0;
    end
    exclude_badHighlightCorr = 1;

    th_AG = -20; % you can change it to -30 to exclude more of the poor quality ROIs.
    th_srt_val = 4150;
    th_smallROI = 15;
    th_shortDecayTau = 200;
    th_badTempCorr = .4;
    th_badHighlightCorr = .4; % .5;

    [badROIs01, bad_EP_AG_size_tau_tempCorr_hiLight_hiLightDB, val_EP_AG_size_tau_tempCorr_hiLight_hiLightDB, th_EP_AG_size_tau_tempCorr_hiLight_hiLightDB] = ...
        findBadROIs(mouse, imagingFolder, mdfFileNumber, fixed_th_srt_val, savebadROIs01, exclude_badHighlightCorr,evalBadRes, th_AG, th_srt_val, th_smallROI, th_shortDecayTau, th_badTempCorr, th_badHighlightCorr);
%     clearvars -except mouse imagingFolder mdfFileNumber days


    %% inhibit excit identification 

    close all

    saveInhibitRois = 1;
    assessClass_unsure_inh_excit = [0,0,0]; %[1 1 1]; %[1,1,0]; % whether to assess unsure, inhibit, excit neuron classes. % you will go through unsure, inhibit and excit ROIs one by one. (if keyEval is 1, you can change class, if 0, you will only look at the images)
    keyEval = 1; % if 1, you can change class of neurons using key presses.
    manThSet = 0; % if 1, you will set the threshold for identifying inhibit neurons by looking at the roi2surr_Sig values.
    % Be very carful if setting keyEval to 1: Linux hangs with getKey if you click anywhere, just use the keyboard keys! % if 0 you will simply go though ROIs one by one, otherwise it will go to getKey and you will be able to change neural classification.

    identifInh = 1; % if 0, only bleedthrough-corrected ch1 image will be created, but no inhibitory neurons will be identified.
    do2dGauss = 0; % Do 2D gaussian fitting on ch1 images of gcamp ROIs for ROIs identified by your measure as unsure.

    inhibit_excit_prep(mouse, imagingFolder, mdfFileNumber, saveInhibitRois, assessClass_unsure_inh_excit, keyEval, manThSet, identifInh, do2dGauss)


    %%%%%% Merge gcamp and tdtomato channel, also mark identified inhibitory neurons
    savefigs = 1; 
    removeBadA = 0; % only used when plotA is 1; if removeBadA = 1, green channel will only show good quality ROIs, otherwise it will show all ROIs.
    for plotA = [0,1] % if 1, green channel will be ROIs identified in A; if 0, it will be the average image of green channel.
        % qhg = .98;
        [rgImg, gcampImg, tdTomatoImg] = inhibit_gcamp_merge(mouse, imagingFolder, mdfFileNumber, savefigs, plotA, removeBadA); %, qhg
    end

    diary off
    
end


%% imaging_prep_analysis

if doPost

    %%% Start a diary file
    nowStr = datestr(now, 'yymmdd-HHMMSS');
    dn = ['diary_post_',date_major, '_',nowStr];
    diary(dn)
    
    close all
    % best is to set the 2 vars below to 0 so u get times of events for all trials; later decide which ones to set to nan.
    rmvTrsStimRateChanged = 0; % if 1, early-go-tone trials w stimRate categ different before and after go tone, will be excluded.
    do = 1; % set to 1 when first evaluating a session, to get plots and save figs and vars.

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
       = imaging_prep_analysis(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength, save_aligned_traces, savefigs, rmvTrsStimRateChanged, thbeg);

   diary off
   
end


