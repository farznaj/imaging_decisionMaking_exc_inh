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
mouse = 'fni17';
imagingFolder = '150918';
mdfFileNumber = [1];  % 3; %1; % or tif major

close all, clearvars -except mouse imagingFolder mdfFileNumber

% find the index of a particular day
d = '150826';
find(~cellfun(@isempty, strfind(days, d)))
%}


%% fni16

% later days
mouse = 'fni16';
days = {'150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2',...
    '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'};

% earlier days
mouse = 'fni16';
days = {'150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150824_1-2',...
    '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', ...
    '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1',...
    '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3',...
    '150928_1-2', '150929_1-2'}; % , '150914_1-2' : dont analyze!


% all days
mouse = 'fni16';
days = {'150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150824_1-2',...
    '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', ...
    '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1',...
    '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3',...
    '150928_1-2', '150929_1-2', ...
    '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2',...
    '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'}; % , '150914_1-2' : dont analyze!

% the following days dont have enough trials...
% days = {'150807_1', '150814_1'}; % 150807 was a different spot than 0814-0819. And 0814-0819 were different from the rest.


%% fni17

% later days
mouse = 'fni17';
days = {'151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', ...
    '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', ...
    '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1',   '151102_1-2'};

% earlier days
mouse = 'fni17';
days = {'150820_1', '150821_1', '150824_1', '150825_1', '150826_1', '150827_1', '150828_1', ...
    '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', ...
    '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', ...
    '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', ...
    '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1'};

% all days
mouse = 'fni17';
days = {'150820_1', '150821_1', '150824_1', '150825_1', '150826_1', '150827_1', '150828_1', ...
    '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', ...
    '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', ...
    '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', ...
    '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', ...
    '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', ...
    '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', ...
    '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1',   '151102_1-2'};


% {'150814_2', '150817_1', '150818_1', '150819_1-2'} % only spots that are the same as the rest of the days
days = {'150814_1', '150814_2', '150817_1', '150818_1', '150819_1-2'}; % 0813 and 0814 were on a different spot than the rest
% '150813_1' : not enough trials


% Below is written on 01/09/2019: these
% days had long mscanLag trials that were not among trs2rmv. I updated
% trs2rmv in imfilename.
% mouse = 'fni17';
% days = {'151010_1'}; 


%% fni18

mouse = 'fni18';
days = {'151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'};


%% fni19

% later days
mouse = 'fni19';
days = {'150922_1', '150923_1', '150924_1-2', '150925_1-2', ...
    '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', ...
    '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3',...
    '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', ...
    '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', ...
    '151028_1-2', '151029_1-2-3', '151101_1'};


% earlier days
mouse = 'fni19';
days = {'150901_1', '150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1'};


% all days
mouse = 'fni19';
days = {'150901_1', '150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1',...
    '150922_1', '150923_1', '150924_1-2', '150925_1-2', ...
    '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', ...
    '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3',...
    '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', ...
    '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', ...
    '151028_1-2', '151029_1-2-3', '151101_1'};


% Below is written on 01/09/2019: these
% days had long mscanLag trials that were not among trs2rmv. I updated
% trs2rmv in imfilename.
% mouse = 'fni19';
% days = {'150924_1-2', '151005_1-2', '151016_1', '151029_1-2-3'}; 


%% Run imaging_postproc for each day

setFrdrops = 0; % set to 1 write after cnmf, so if needed you reran cnmf
bad_mask_inh = 0; % if 1, badROIs, mask, and inh/exc will be set (ie almost all except for imaging_prep_analysis)
doPost = 1; % if 1 imaging_prep_analysis will be run.

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
        me = ME;
        disp(ME)
        celldisp({me.stack.name}), {me.stack.line}
        cd(md), save(['err_', imagingFolder], 'me')
        diary off
    end
    %%%
    close all
%         clearvars -except me mouse days setFrdrops bad_mask_inh doPost md iday mice
end
% end


%% Publish figures in a summary pdf file.

savedir0 = fullfile('~/Dropbox/ChurchlandLab/Farzaneh_Gamal/postprop_sum',mouse);

if ~exist('savedir0', 'dir')
    mkdir(savedir0)
end

% tic
for iday = 1:length(days)

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
%     try
        publish('/home/farznaj/Documents/trial_history/imaging/imaging_postProc_sum.m', 'format', 'pdf')

        close all

        f = ls('~/Documents/trial_history/imaging/html/*_sum*');
        [~,f2,f3] = fileparts(f);
        savedir = fullfile(savedir0, [date_major, '_', f2,f3]);

        movefile(f, savedir) % '~/Documents/trial_history/imaging/html/*_sum*'

%             clearvars -except mouse days savedir0

        %         savedir = fullfile(savedir0, date_major);
        %         if ~exist(savedir, 'dir')
        %             mkdir(savedir)
        %         end

%     catch ME
%         disp(ME)
%     end
end


% t = toc
%{
a = dir;
a(~[a.isdir])=[];
aa = {a.name};
for i=3:length(aa)
    movefile(fullfile(aa{i}, 'imaging_postProc_sum.pdf'), [aa{i}, '_imaging_postProc_sum.pdf'])
end
%}






%% Add missing fields to alldata

doclean = 1;
defaultHelpedTrs = 0;
saveHelpedTrs = 1;


%%%%%%
for iday = 1:length(days)
    
    disp('__________________________________________________________________')
    dn = simpleTokenize(days{iday}, '_');
    
    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
    
    fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})
    
    
    %%%    
    % set filenames
    [alldata_fileNames, ~] = setBehavFileNames(mouse, {datestr(datenum(imagingFolder, 'yymmdd'))});
    % sort it
    [~,fn] = fileparts(alldata_fileNames{1});
    a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
    [~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
    alldata_fileNames = alldata_fileNames(isf);

    % load the one corresponding to mdffilenumber.
    [data, trials_per_session] = loadBehavData(alldata_fileNames(mdfFileNumber), defaultHelpedTrs, saveHelpedTrs, doclean);

end






%% Remove weights var from diffNumNeurons svm files

tic
for mice = {'fni16','fni17','fni18','fni19'}
    
    mouse = mice{1};
    fprintf('Analyzing mouse %s\n', mouse)
    
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


    %%
    for iday = 1:length(days)

        disp('__________________________________________________________________')
        dn = simpleTokenize(days{iday}, '_');

        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));

        fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})

        %%
        try
            signalCh = 2; % because you get A from channel 2, I think this should be always 2.
            pnev2load = [];
            [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
            % [pd, pnev_n] = fileparts(pnevFileName);
            [md,date_major] = fileparts(imfilename);
            sv = fullfile(md,'svm');
%             cd(sv) 

            a = dir('diffNumNs*');
            for i = 1:length(a)
                m = matfile(a(i).name);
                mm = fieldnames(m);
                r = regexp(mm, 'wAllC_nN_*');
                if cell2mat(r) % check to see if weights exit
                    f = find(cellfun(@(x)~isempty(x), r)); %index of wAllc                
                    fprintf('removing %s\n', mm{f})
                    rmvar(a(i).name, mm{f})
                end
            end

        catch ME
            me = ME;
            disp(ME)
            celldisp({me.stack.name}), {me.stack.line}
            cd(sv), save(['err_', imagingFolder], 'me')
            diary off
        end
    end

end
t = toc

