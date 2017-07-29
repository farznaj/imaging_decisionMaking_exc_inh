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
close all
mouse = 'fni19';
imagingFolder = '151029';
mdfFileNumber = [1,2,3];  % 3; %1; % or tif major

close all, clearvars -except mouse imagingFolder mdfFileNumber
%}


%%
mouse = 'fni16';
days = {'150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2',...
    '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'};

%%
mouse = 'fni17';
days = {'151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', ...
        '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', ...
        '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1',   '151102_1-2'};
    

mouse = 'fni17';
days = {'150820_1', '150821_1', '150824_1', '150825_1', '150826_1', '150827_1', '150828_1', ...
    '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', ...
    '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', ...
    '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', ...
    '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1'};

%%
mouse = 'fni18';
days = {'151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'};

%%
mouse = 'fni19';
days = {'150922_1', '150923_1', '150924_1-2', '150925_1-2', ...
    '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', ...
    '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3',...
    '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', ...
    '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', ...
    '151028_1-2', '151029_1-2-3', '151101_1'};

%% Run imaging_postproc for each day

% tic
for iday = 1:length(days)
    
    disp('__________________________________________________________________')
    dn = simpleTokenize(days{iday}, '_');
    
    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));   

    fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})    
    %%%
    try
        imaging_postproc    
    catch ME
        disp(ME)
    end
    %%%
    close all
    clearvars -except mouse days    
end
% t = toc


%% Publish figures in a summary pdf file.

savedir0 = fullfile('~/Dropbox/ChurchlandLab/Farzaneh_Gamal/postprop_sum',mouse);

% tic
for iday = 3 %1:length(days)
    
    disp('__________________________________________________________________')
    dn = simpleTokenize(days{iday}, '_');
    
    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));   

    fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})   
    
    %%%
    signalCh = 2; % because you get A from channel 2, I think this should be always 2.
    pnev2load = [];
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    disp(pnevFileName)
    [pd, date_major] = fileparts(imfilename);
    figd = fullfile(pd, 'figs');     % cd(figd) %%% copyfile(fullfile('/home/farznaj/Documents/trial_history/imaging','imaging_postProc_html.m'),'.','f')

    %%
    try
        publish('/home/farznaj/Documents/trial_history/imaging/imaging_postProc_sum.m', 'format', 'pdf')

        close all

        savedir = fullfile(savedir0, date_major);
        if ~exist(savedir, 'dir')
            mkdir(savedir)
        end
        movefile('~/Documents/trial_history/imaging/html/*_sum*', savedir)

        clearvars -except mouse days savedir0
    catch ME
        disp(ME)
    end
end
% t = toc

    