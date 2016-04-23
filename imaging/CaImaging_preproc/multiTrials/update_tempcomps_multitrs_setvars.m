function [cs_frtrs, Nnan] = update_tempcomps_multitrs_setvars(mousename, imagingFolder, mdfFileNumber, allTifMinors, tifMinor)
% cs_frtrs: cumsum of #frames across trials.
% Nnan: Nnan, ie lengths of inter-trial intervals (using alldata) that have been removed
%
% [cs_frtrs, Nnan] = update_tempcomps_multitrs_setvars(mousename, imagingFolder, mdfFileNumber);

if ~exist('allTifMinors', 'var')
    allTifMinors = 1;
end

if ~exist('tifMinor', 'var')
    tifMinor = [];
end


%% Compute cumsum of #frames across trials.

% tifMinor = 4;
% [cs_frtrs, firstFullTrial, lastFullTrial, frs_trFirstInMov, frs_trEndInMov, nImagingTrsTotal] = framesPerTrialMovie(mousename, imagingFolder, mdfFileNumber); %, tifMinor);

if allTifMinors % analyze all tif minors.
    [cs_frtrs, firstFullTrial, lastFullTrial, frs_trFirstInMov, frs_trEndInMov, nImagingTrsTotal] = framesPerTrialMovie(mousename, imagingFolder, mdfFileNumber);

elseif length(tifMinor)==1
    [cs_frtrs, firstFullTrial, lastFullTrial, frs_trFirstInMov, frs_trEndInMov, nImagingTrsTotal] = framesPerTrialMovie(mousename, imagingFolder, mdfFileNumber, tifMinor);
    
elseif ismember(1, diff(tifMinor)) % at least 2 movies are consecutive. %length(unique(diff(tifMinor)))==1 
    if all(unique(diff(tifMinor))==1) % all are consecutive movies
        cs_frtrs = framesPerTrialMovie(mousename, imagingFolder, mdfFileNumber, tifMinor(1), tifMinor(end));

    else % some movies are consecutive and some are not.
        error('Needs work to incorporate this case! Use a combination of consecutive and discontinous cases.')
        % do a combination of consecutive and discontinous cases written above.
    end

else % tifMinors are all discontiuous movies

    cs_frtrs = 0;
    for itm = tifMinor
        cs_frtrs_now = framesPerTrialMovie(mousename, imagingFolder, mdfFileNumber, itm);
        cs_frtrs = [cs_frtrs  cs_frtrs(end) + cs_frtrs_now(2:end)];

    end
end
Ntrs = length(cs_frtrs)-1;
fprintf('# imaging trials: %d\n', Ntrs)
fprintf('Remember last element could be a partial trial:\nif bcontrol was aborted at the middle of the trial,\nno behavioral states exit but some recorded frames will exist!!\n')


%% Compute Nnan, ie lengths of inter-trial intervals (using alldata) that have been removed

% Define upper limit for ITI length = # frames (Dt) to reach .01 of peak value of a spike.
% assumption about decay tau : ~700ms when gd=.955 (tau_d = frameLength * -1/log(.955))
% nan_upperlimit = ceil(log(.01)/log(.955));
nan_upperlimit = ceil(log(.01)/log(.97)); % gd=.97 corresponds to tau_d of 1000ms


frameLength = 1000/30.9; % msec.

% load alldata
% set filenames
[alldata_fileNames, ~] = setBehavFileNames(mousename, {datestr(datenum(imagingFolder, 'yymmdd'))});
% sort it
[~,fn] = fileparts(alldata_fileNames{1});
a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
[~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
alldata_fileNames = alldata_fileNames(isf);
% load the one corresponding to mdfFileNumber.
[all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber)); %, defaultHelpedTrs, saveHelpedTrs); % it removes the last trial too.
fprintf('# behavioral trials: %d\n', length(all_data))

iti_noscan = itiSet(all_data)'; % ms
iti_noscan(1) = []; % so iti_noscan(i) gives ITI following trial i.
% iti_noscan = iti_noscan(1:Ntrs-1); % get ITIs corresponding to imaging trials. (we dont care about ITI following last imaging trials.)

nFrames_iti = round(iti_noscan/frameLength); % nFrames between trials (during iti) that were not imaged.

% if Ntrs > length(all_data) % use nan_upperlimit as an upper estimate of the ITI for the trials that exist in imaging but not behavior.
if nImagingTrsTotal > length(all_data)    
    warning('Check this!!')
    % this is a likely scenario: when bcontrol stops at the middle of a
    % trial, we loose that trial in all_data, but imaged frames (up to the
    % point of trial abortion) remain. Hence Ntrs will be 1 element longer
    % than all_data.
%     warning(sprintf('%i trial was in imaging but not behavioral data. Using nan_upper_limit as ITI!', Ntrs - length(all_data)))
%     nFrames_iti = [nFrames_iti nan_upperlimit*ones(1, Ntrs - length(all_data))];
    warning(sprintf('%i trial was in imaging but not behavioral data. Using nan_upper_limit as ITI!', nImagingTrsTotal - length(all_data)))
    nFrames_iti = [nFrames_iti nan_upperlimit*ones(1, nImagingTrsTotal - length(all_data))];
    
    if nImagingTrsTotal - length(all_data) > 1
        error('What is going on?')
    end
end


if frs_trFirstInMov > 0
    nFrames_iti = nFrames_iti([firstFullTrial-1, firstFullTrial : lastFullTrial]); % 100:107 are trials that correspond to movie 4. 99 was partially recorded at the beginning.
else
    nFrames_iti = nFrames_iti([firstFullTrial : lastFullTrial]);
end


if ~allTifMinors && length(tifMinor)>1
    error('Setting nFrames_iti needs work if only some of the TIF movies are used! See the commented codes.')    
end



% Take care of upper limit for nans.
Nnan = nFrames_iti;      % lengths of inter-trial intervals that have been removed
Nnan(Nnan > nan_upperlimit) = nan_upperlimit;


% figure; plot(nFrames_iti)
% hold on; plot(Nnan)


