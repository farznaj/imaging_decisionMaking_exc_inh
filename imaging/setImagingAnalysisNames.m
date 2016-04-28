function [imfilename, pnevFileName, tifFold, date_major] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh, pnev2load)
% [imfilename, pnevFileName, tifFold, date_major] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh)
% E.g.
% mousename = 'fni17';
% imagingFolder = '151102'; % '151021';
% mdfFileNumber = 1; % or tif major
% signalCh = 2;

if ~exist('pnev2load', 'var') || isempty(pnev2load)
    pnev2load = 1; % use the most recent file.
end


%%
if isempty(strfind(pwd, 'gamalamin')) % Farzaneh
    if ismac
        dataPath = '/Users/Farzaneh/Desktop/Farzaneh/data'; % macbook
    elseif isunix
        dataPath = '/sonas-hs/churchland/nlsas/data/data'; % server
    elseif ispc
        dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % lab PC
    end
else % Gamal
    dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data';
end


tifFold = fullfile(dataPath, mousename, 'imaging', imagingFolder);
date_major = sprintf('%s_%03d', imagingFolder(1:6), mdfFileNumber);
imfilename = fullfile(tifFold, date_major);

pnevFileName = [date_major, '_ch', num2str(signalCh),'-Pnev*'];
pnevFileName = dir(fullfile(tifFold, pnevFileName));

if isempty(pnevFileName)
    fprintf('No Pnev file was found!\n')
    pnevFileName = '';
else
    % in case there are a few pnev files, choose which one you want!
    [~,i] = sort([pnevFileName.datenum], 'descend');
    disp({pnevFileName(i).name}')
    pnevFileName = pnevFileName(i(pnev2load)).name;
    pnevFileName = fullfile(tifFold, pnevFileName);
end

