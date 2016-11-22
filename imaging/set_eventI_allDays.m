% function eventI_allDays = set_eventI_allDays(days, trialHistAnalysis)
% Compute eventI for all days 

%{
trialHistAnalysis = 1;

mouse = 'fni17';
days = {'151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', ...
    '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', ...
    '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'};
%}

%%
eventI_allDays = nan(1,length(days));

for iday = 1:length(days)
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
    
    
    if trialHistAnalysis==1
        load(postName, 'stimAl_allTrs')
        stimAl = stimAl_allTrs;
    else
        load(postName, 'stimAl_noEarlyDec')
        stimAl = stimAl_noEarlyDec;
    end
    
    eventI_stimOn = stimAl.eventI;

    eventI_allDays(iday) = eventI_stimOn;
end

