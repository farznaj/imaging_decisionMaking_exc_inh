function newBadROIs = findBadROIs_strict(mouse, imagingFolder, mdfFileNumber);
% Redefine good ROIs out of ~badROIs01 using stricter thresholds for
% fitness and highlightCorr.
%
% Output: newBadROIs is of size sum(~badROIs01) x 1, and equals 1 for
% badROIs using stricter thresholds.
%
% Below for example includes stim-aligned traces for the newly defined good ROIs, using strict thresholds in this script:
% newBadROIs = findBadROIs_strict(mouse, imagingFolder, mdfFileNumber);
% tracesHiQual = stimAl_allTrs.traces(:,~newBadROIs, :);


%% Change these:
%{
mouse = 'fni17';
imagingFolder = '151010'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1];  % 3; %1; % or tif major
%}

%  Define more strict thresholds for fitness and highlightCorr
th_AG = -30; % normally -20, make it more strict.
th_badHighlightCorr = .5; % normally we just use rval_space w th = .4; our corr measure is more strict than EP and we change th to .4 to make it more strict.


%% Set .mat file names

signalCh = 2;
pnev2load = []; %7 %4 % what pnev file to load (index based on sort from the latest pnev vile). Set [] to load the latest one.
postNProvided = 1; % whether the directory cotains the postFile or not. if it does, then mat file names will be set using postFile, otherwise using pnevFile.
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load, postNProvided);
[pd, pnev_n] = fileparts(pnevFileName);
postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));


%% Load AG's fitness measure and highLighCorr measure

load(pnevFileName, 'highlightCorrROI')
load(moreName, 'fitness', 'badROIs01')


%% Redefine badROIs (newBadROIs) using strict thresholds

fitNow = fitness(~badROIs01)';
hlNow = highlightCorrROI(~badROIs01)';

fitStrict = fitNow > th_AG;
hlStrict = hlNow < th_badHighlightCorr;

newBadROIs = fitStrict | hlStrict;
newBadROIs_hl = hlStrict;
newBadROIs_ag = fitStrict;

n = sum(newBadROIs);
cprintf('red', '%d ROIs to be excluded out of %d ROIs using strict thresholds!\n', n, sum(~badROIs01))





