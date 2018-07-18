mice = {'fni16','fni17','fni18','fni19'};

% close all
% best is to set the 2 vars below to 0 so u get times of events for all trials; later decide which ones to set to nan.
rmvTrsStimRateChanged = 0; % if 1, early-go-tone trials w stimRate categ different before and after go tone, will be excluded.
do = 0; % set to 1 when first evaluating a session, to get plots and save figs and vars.

normalizeSpikes = 1; % if 1, spikes trace of each neuron will be normalized by its max.
warning('Note you have set normalizeSpikes to 1!!!')

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



%%
for im = 1:length(mice)
    
    mouse = mice{im};
    fprintf('==============================  Mouse %s ==============================\n\n', mouse)
    
    %%% Set days for each mouse

    if strcmp(mouse, 'fni16')
        days = {'150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'}; %'150914_1-2' : don't analyze!
    elseif strcmp(mouse, 'fni17')
        days = {'150814_1', '150817_1', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};
    elseif strcmp(mouse, 'fni18')
        days = {'151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'}; % alldays
    elseif strcmp(mouse, 'fni19')    
        days = {'150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'};
    end    
    % fni16:         % you use the ep of each day to compute ROC ave for setting ROC histograms.        %     days = {'150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'};
    % fni17:         % ep_ms = [800, 1100]; % window to compute ROC distributions ... for fni17 you used [800 1100] to compute svm which does not include choice... this is not the case for fni16.         %     days = {'151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};
    
    
    %% Align wheel revolution and lick traces for each day

    for iday = 1:length(days)
        
        thbeg = 5; % n initial trials to exclude.
        if strcmp(mouse, 'fni19') && strcmp(imagingFolder,'150918')
            thbeg = 7;
        end
        
        dn = simpleTokenize(days{iday}, '_');
        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
        fprintf('\n_________________  %s, day %d/%d (%s, sessions %s)  _________________\n', mouse, iday, length(days), imagingFolder, dn{2})
    
        tracesAlign_wheelRev_lick(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength, save_aligned_traces, savefigs, rmvTrsStimRateChanged, thbeg);

    end
end


