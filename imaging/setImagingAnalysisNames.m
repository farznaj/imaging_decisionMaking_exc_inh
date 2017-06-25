function [imfilename, pnevFileName, tifFold, date_major] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh, pnev2load, postNProvided)
% [imfilename, pnevFileName, tifFold, date_major] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh)
% E.g.
% mousename = 'fni17';
% imagingFolder = '151102'; % '151021';
% mdfFileNumber = 1; % or tif major
% signalCh = 2;

if ~exist('pnev2load', 'var') || isempty(pnev2load)
    pnev2load = 1; % use the most recent file.
end

if ~exist('postNProvided', 'var')
    postNProvided = 0;
end


%%
if isempty(strfind(pwd, 'gamalamin')) % Farzaneh
    if ismac
        dataPath = '/Users/Farzaneh/Desktop/Farzaneh/data'; % macbook
    elseif isunix
        if isempty(strfind(pwd, 'grid')) % 'sonas')) % Unix in the office
            dataPath = '~/Shares/Churchland/data';
            altDataPath = '~/Shares/Churchland_hpc_home/space_managed_data'; % the new space-managed server (wos, to which data is migrated from grid)
        else % server
            dataPath = '/sonas-hs/churchland/nlsas/data/data';
            altDataPath = '/sonas-hs/churchland/hpc/home/space_managed_data';
        end
    elseif ispc
        dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % Office PC
    end
else % Gamal
    dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data';
end


tifFold = fullfile(dataPath, mousename, 'imaging', imagingFolder);

if ~exist(tifFold, 'dir') 
    if exist('altDataPath', 'var')
        tifFold = fullfile(altDataPath, mousename, 'imaging', imagingFolder);
    else
        error('Data directory does not exist!')
    end
end

% date_major = sprintf('%s_%03d', imagingFolder(1:6), mdfFileNumber);
r = repmat('%03d-', 1, length(mdfFileNumber));
r(end) = [];
date_major = sprintf(['%s_', r], imagingFolder, mdfFileNumber);

imfilename = fullfile(tifFold, date_major);

if exist('signalCh', 'var')
    if postNProvided
        pnevFileName = ['post_', date_major, '_ch', num2str(signalCh),'-Pnev*'];
    else
        pnevFileName = [date_major, '_ch', num2str(signalCh),'-Pnev*'];
    end
    
    pnevFileName = dir(fullfile(tifFold, pnevFileName));
    
    if isempty(pnevFileName)
        fprintf('No Pnev file was found!\n')
        pnevFileName = '';
    else
        
        % in case there are a few pnev files, choose which one you want!
        [~,i] = sort([pnevFileName.datenum], 'descend');
        disp({pnevFileName(i).name}')
        pnevFileName = pnevFileName(i(pnev2load)).name;
        if postNProvided
            pnevFileName = pnevFileName(6:end);
        end
        pnevFileName = fullfile(tifFold, pnevFileName);
%         disp(pnevFileName)
    end
    
else
    pnevFileName = '';
end

