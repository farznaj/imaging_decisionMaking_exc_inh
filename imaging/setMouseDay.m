% Right after you are done with preproc on the cluster, run the following scripts:
% - plotMotionCorrOuts
% - plotEftyVarsMean (if needed follow by setPmtOffFrames to set pmtOffFrames and by findTrsWithMissingFrames to set frame-dropped trials. In this latter case you will need to rerun CNMF!): for a quick evaluation of the traces and spotting any potential frame drops, etc
% - eval_comp_main on python (to save outputs of Andrea's evaluation of components in a mat file named more_pnevFile)
% - set_mask_CC
% - findBadROIs
% - inhibit_excit_prep
% - imaging_prep_analysis (calls set_aligned_traces... you will need its outputs)

%%
%{
mouse = 'fni18';
imagingFolder = '151217';
mdfFileNumber = [1,2];  % 3; %1; % or tif major
%}

for iday = 1:length(days)
    
    disp('__________________________________________________________________')
    dn = simpleTokenize(days{iday}, '_');
    
    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));   

    fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})   
    
    %{
    signalCh = 2; % because you get A from channel 2, I think this should be always 2.
    pnev2load = [];

    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    [pd,pnev_n] = fileparts(pnevFileName);
    disp(pnev_n)
    cd(fileparts(imfilename))

    moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));

    load(moreName, 'slope_common_pix')    
    disp(slope_common_pix)
    
    cd figs
    open('red_green.fig')
    set(gcf,'name', num2str(slope_common_pix))
    pause
    %}
% end
    
    
    %%    
    savebadROIs01 = 1; % if 1, badROIs01 will be appended to more_pnevFile
    evalBadRes = 0; % plot figures to evaluate the results

    if ismember(mouse, {'fni16', 'fni17'})
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

%     clearvars -except mouse imagingFolder mdfFileNumber


    %% inhibit excit identification 

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
    for plotA = [0,1]; % if 1, green channel will be ROIs identified in A; if 0, it will be the average image of green channel.
        % qhg = .98;
        [rgImg, gcampImg, tdTomatoImg] = inhibit_gcamp_merge(mouse, imagingFolder, mdfFileNumber, savefigs, plotA, removeBadA); %, qhg
    end
    
    %%
    close all
    clearvars -except mouse days
    
end


%%
mouse = 'fni18';
days = {'151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'};


%%
mouse = 'fni17';
days = {'151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', ...
        '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', ...
        '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1',   '151102_1-2'};
    
% '151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', ...
%     '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', ...
%     '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'    


%%
mouse = 'fni16';
days = {'150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2',...
    '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'};


%%
%{
rmv_timeGoTone_if_stimOffset_aft_goTone = 0; % if 1, trials with stimOffset after goTone will be removed from timeGoTone (ie any analyses that aligns trials on the go tone)
rmv_time1stSide_if_stimOffset_aft_1stSide = 0; % if 1, trials with stimOffset after 1stSideTry will be removed from time1stSideTry (ie any analyses that aligns trials on the 1stSideTry)

normalizeSpikes = 1; % if 1, spikes trace of each neuron will be normalized by its max.

% set the following vars to 1 when first evaluating a session.
evaluateEftyOuts = 0;
compareManual = 0; % compare results with manual ROI extraction

plot_ave_noTrGroup = 0; % Set to 1 when analyzing a session for the 1st time. Plots average imaging traces across all neurons and all trials aligned on particular trial events. Also plots average lick traces aligned on trial events.

setInhibitExcit = 0; % if 1, inhibitory and excitatory neurons will be set unless inhibitRois is already saved in imfilename (in which case it will be loaded).
plotEftyAC1by1 = 0; % A and C for each component will be plotted 1 by 1 for evaluation of of Efty's results.
frameLength = 1000/30.9; % sec.
%}

% 
% %%
% for iday = 1:length(days)
%     disp('__________________________________________________________________')
%     dn = simpleTokenize(days{iday}, '_');
%     
%     imagingFolder = dn{1};
%     mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));   
% 
%     fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})
%     
%     % Run imaging_prep_analysis
%     imaging_prep_analysis(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength);    
% end




    