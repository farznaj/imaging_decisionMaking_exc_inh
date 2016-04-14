function mdfToTif_setParams(mousename, day, dir_local, mdfFileNumbers, tifInds, deleteMDF, deleteTif, copyToServer)
% mdfToTif_setParams(mousename, day, dir_local, mdfFileNumbers, tifInds, deleteMDF, deleteTif, copyToServer)
%
% Initialize parameters and calls openMDF and mdfToTif to convert a MDF
% file to all its TIF files. This funciton uses MCSX library, which can
% only be run on Windows and MView needs to be installed.
%
% MDF file will be copied from the server to the local computer
% (dir_local\mousename\day\'imaging'), unless it already exists there. TIF
% files (each including 8000 frames) will be written on the local computer.
% Once done, they will be copied to the server (unless copyToServer set to
% 0). MDF and tif files will be deleted from the local computer (unless
% deleteTif and deleteTif set to 0). If copyToServer, deleteTif, deleteMDF
% are not provided they will be set to 1.
%
% INPUTS:
%
% mousename -- string, e.g. 'FNI17'
% day -- string, date, e.g. '151020'
% dir_local -- string, directory on the local computer. fullfile(dir_local,
%       mousename, day, 'imaging') will be where mdf file will be copied
%       and tif files will be created. e.g.
%       'C:\Users\fnajafi\Documents\Data\'
%
% Optional inputs:
% mdfFileNumbers -- numeric array, mdf file numbers to convert, e.g. [1,2].
%       Default: all MDF files in the mousename/day directory.
% tifInds -- numeric array, e.g. tifInds=[1,3] will only create the 1st and
%       3rd TIF files, i.e. frames 1:8000 and 16001:24000 of the MDF file
%       will be converted. If not provided, all MDF frames will be
%       converted.
% deleteMDF -- default: 1, if 0 MDF file wont be deleted from the local
%       computer.
% deleteTif -- default: 1, if 0 TIF files wont be deleted from the local
%       computer.
% copyToServer -- default: 1, if 0 TIF files wont be copied to the server.
%
% (Farzaneh Najafi, Oct 20 2015).


%% Set some parameters.
if ~exist('copyToServer', 'var') || copyToServer~=0
    copyToServer = true;
end

if ~exist('deleteTif', 'var') || deleteTif~=0
    deleteTif = true;
end

if ~exist('deleteMDF', 'var') || deleteMDF~=0
    deleteMDF = true;
end


imdir_server = fullfile('\\sonas-hs.cshl.edu\churchland\data', mousename, 'imaging', day);

a = dir(fullfile(imdir_server, sprintf('%s_*.MDF', day)));
mdfList = {a.name};

if ~exist('mdfToAn', 'var')
    mdfFileNumbers = 1:length(mdfList);
end


%%
for filenumber = mdfFileNumbers
    
    [~, imfilename] = fileparts(mdfList{filenumber});
    % imfilename = sprintf('%s_%03d', day, filenumber);
    
    imdir = fullfile(dir_local, mousename, day, 'imaging');
    if ~exist(imdir, 'dir'), mkdir(imdir), end
    
    
    %% Copy mdf file from server to your local PC.
    fprintf(['\n====================',mousename,'====================\n'])
    
    mdfFileName = mdfList{filenumber};
    % mdfFileName = [imfilename, '.mdf'];
    
    cd(imdir)
    if ~exist(mdfFileName, 'file')
        tic
        s = fullfile(imdir_server, mdfFileName);
        
        fprintf('Copying mdf file %s to the local computer...\n', mdfFileName)
        copyfile(s, imdir)
        
        fprintf('... took %s (hr,min,sec)\n', datestr(toc/86400, 'HH:MM:SS')) % fprintf('... took %.2f sec\n', toc)
    end
    
    
    %% Open the mdf file using MCSX library.
    cd(imdir)
    [mfile, OpenResult] = openMDF(mdfFileName);
    
    if OpenResult~=0
        error('Unable to open MDF file!!')
    end
    
    
    %% Read some params from the mdf file.
    % Set scanning channels
    chS = cell(1,4);
    channelsSaved = [];
    for ch = 1:4
        if isempty(mfile.ReadParameter(sprintf('Scanning Ch %d Name', ch-1)))
            chS{ch} = 'Not Saved';
        else
            chS{ch} = 'Saved';
            channelsSaved = [channelsSaved, ch];
        end
    end
    fprintf('Scanning channels: %s\n', num2str(channelsSaved))
    
    
    % Total number of frames
    frameCount = mfile.ReadParameter('Frame Count');
    frameCount = str2double(frameCount);
    fprintf('Total number of frames = %d\n', frameCount*length(channelsSaved))
    
    
    % Image description
    ImgDescription = sprintf('Creator: %s\n\nChannel 1 Name: %s\n\nChannel 2 Name: %s\n\nChannel 3 Name: %s\n\nChannel 4 Name: %s\n\nChannel 1: %s\n\nChannel 2: %s\n\nChannel 3: %s\n\nChannel 4: %s\n\nScan Mode: %s\n\nObjective: %s\n\nMicrons per pixel: %s\n\nX Frame Offset: %s\n\nY Frame Offset: %s\n\nZ Stack Interval: %s\n\nZ Stack Section Count: %s\n\nZ Stack Average Count: %s\n\nFast Z Stack Interval: %s\n\nRotation Angle: %s\n\nMagnification: %s\n\nX Stage position: %s\n\nY Stage position: %s\n\nZ Stage position: %s\n\nFrame Duration: %s',...
        mfile.ReadParameter('Created By'), mfile.ReadParameter('Scanning Ch 0 Name'), mfile.ReadParameter('Scanning Ch 1 Name'),...
        mfile.ReadParameter('Scanning Ch 2 Name'), mfile.ReadParameter('Scanning Ch 3 Name'),...
        chS{1}, chS{2}, chS{3}, chS{4}, ...
        mfile.ReadParameter('Scan Mode'), mfile.ReadParameter('Objective'), mfile.ReadParameter('Microns Per Pixel'),...
        mfile.ReadParameter('X Frame Offset'), mfile.ReadParameter('Y Frame Offset'),...
        mfile.ReadParameter('Z- Interval'), mfile.ReadParameter('Section Count'), mfile.ReadParameter('Averaging Count'),...
        mfile.ReadParameter('Fast Stack z- Interval'),...
        mfile.ReadParameter('Rotation'), mfile.ReadParameter('Magnification'),...
        mfile.ReadParameter('X Position'), mfile.ReadParameter('Y Position'), mfile.ReadParameter('Z Position'),...
        mfile.ReadParameter('Frame Duration (s)'));
    
    
    %% Set input params for mdfToTif function (tifList, and frameArr).
    numFramesEachTif = 8000/length(channelsSaved); % each tif file can have ~8000 frames.
    tifList = cell(1, ceil(frameCount/numFramesEachTif));
    for itif = 1:length(tifList)
        tifList{itif} = sprintf('%s_%02d', imfilename, itif);
    end
    % showcell(tifList)
    
    frameArr = 1:numFramesEachTif:numFramesEachTif*length(tifList);
    frameArr = [frameArr frameCount+1];
    
    
    %% Convert mdf file to all its tif files.
    if ~exist('tifInds', 'var')
        tifInds = 1:length(tifList);
    end
    
    for itif = tifInds
        
        tifName = tifList{itif};
        framenums = frameArr(itif) : frameArr(itif+1)-1;
        
        fprintf('\n\nWriting tif file %s (%d/%d)\n', tifName, itif, length(tifList));
        fprintf('Frames %d to %d of the MDF file from %d channels\n', framenums(1), framenums(end), length(channelsSaved))
        
        tic
        mdfToTif(mfile, tifName, framenums, channelsSaved, ImgDescription)
        fprintf('\n... took %s (hr,min,sec)\n', datestr(toc/86400, 'HH:MM:SS.FFF')) % fprintf('... took %.2f sec\n', toc)
        
    end
    
    
    %% copy TIF files to the server.
    
    if copyToServer
        tifFileName = [imfilename, '_*', '.TIF'];
        s = fullfile(imdir, tifFileName);
        copyfile(s, imdir_server)
    end
    
    %% Delete tif and mdf files from the local computer
    
    if deleteTif
        cd(imdir)
        delete(tifFileName)
    end
    
    if deleteMDF
        close(1)
        delete(mdfFileName)
    end
    
    %%
    fprintf('\n=============%s done!  %s==============\n', mousename, datestr(now))
    
end


