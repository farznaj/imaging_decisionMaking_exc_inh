mouse = 'fni17';
imagingFolder = '150918';
mdfFileNumber = [1];  % 3; %1; % or tif major


%%
for iday = 18 %1:length(days)

    disp('__________________________________________________________________')
    dn = simpleTokenize(days{iday}, '_');

    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));

    fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})

    signalCh = 2; % because you get A from channel 2, I think this should be always 2.
    pnev2load = [];
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    [pd, pnev_n] = fileparts(pnevFileName);
    moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
    postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));

        
    %% load behavioral data

    [alldata_fileNames, ~] = setBehavFileNames(mouse, {datestr(datenum(imagingFolder, 'yymmdd'))});
    % sort it
    [~,fn] = fileparts(alldata_fileNames{1});
    a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
    [~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
    alldata_fileNames = alldata_fileNames(isf);
    % load the one corresponding to mdffilenumber.
    [all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber)); % , defaultHelpedTrs, saveHelpedTrs); % it removes the last trial too.
    fprintf('Total number of behavioral trials: %d\n', length(all_data))



    %% build regressors
    % time (offset):


    %%
    
    
end

