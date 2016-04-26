function [cs_frtrs, firstFullTrial, lastFullTrial, frs_trFirstInMov, frs_trEndInMov, nImagingTrsTotal] = framesPerTrialMovie(mousename, imagingFolder, mdfFileNumber, movieSt, movieEn)
% cumsum of frame numbers corresponding to each trial recorded in the entire movie of a particular tif movie.
% In the case of analyzing all movies, cs_frtrs is exactly cumsum of
% framesPerTrial, but it will also include the last frames that were
% recorded while bcontrol was aborted.


%%
if ~exist('movieSt', 'var') % cs_frtrs will be computed for the entire movie (not a specific tif minor).
    movieSt = [];
end

if ~exist('movieEn', 'var')
    movieEn = movieSt;
end


%%
if isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data';
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % FN
end
tifFold = fullfile(dataPath, mousename, 'imaging', imagingFolder);
date_major = sprintf('%s_%03d', imagingFolder, mdfFileNumber);


%% Set the entire number of recorded frames

load(fullfile(tifFold, date_major), 'badFrames', 'framesPerTrial')
numRecFrs = length(badFrames{1});

framesPerTrialNoNaN = framesPerTrial(~isnan(framesPerTrial));
cs_frtrs_all = unique([0 cumsum(framesPerTrialNoNaN) numRecFrs]); % if some frames were recorded without a trial, add those frames too.


%% All movies: 

%     nfrs_perTr_inMovieStEn = [framesPerTrialNoNaN , numRecFrs - sum(framesPerTrialNoNaN)];
cs_frtrs = cs_frtrs_all;

cc = cumsum(framesPerTrialNoNaN);
firstFullTrial = 1; 
lastFullTrial = length(framesPerTrialNoNaN);
frs_trFirstInMov = 0;
frs_trEndInMov = numRecFrs - cc(end);

if frs_trEndInMov>0
    nImagingTrsTotal = length(framesPerTrialNoNaN)+1; % includes the last partial trial that the imaging frames exist but its empty in alldata bc bcontrol was aborted at the middle of this last trial.
else
    nImagingTrsTotal = length(framesPerTrialNoNaN);
end
    

%% Analyze some of the movies:

if ~isempty(movieSt)   
    
    % Compute number of frames recorded for each tif movie (tifMinor).
    
    % load DFToutputs of the first tif minor, and use its size to figure
    % out how many frames were saved per Tif file (except for the last
    % tif file that includes whatever frame is remained).
    a = dir(fullfile(tifFold, [date_major, '_*01.mat']));
    load(fullfile(tifFold, a.name), 'DFToutputs')
    nFramesPerMovie_est = size(DFToutputs{find(cellfun(@(x) ~isempty(x), DFToutputs),1)} , 1);
    
    % set the frames corresponding to tifMinor
    cs = [0:nFramesPerMovie_est:numRecFrs numRecFrs]; % [0 cumsum(nFramesPerMovie)];
    nFramesPerMovie = diff(cs);
    
    
    %% Find the frame numbers corresponding to each trial recorded in a particular movie.
    
    % find the number of frames for each trial recorded from the start of
    % movieSt to the end of movieEn.
    
    stFr = cs(movieSt)+1; % 1st frame of movie 4 in the concatenated movie.
    ff = cs_frtrs_all - stFr;
    trMovStarted = find(ff >= 0, 1) - 1; % in what trial movie 4 started.
    frs_trFirstInMov = cs_frtrs_all(trMovStarted+1) +1 - stFr; % num frames of trial trMovStarted that were recorded in movieSt. (it is perhaps the last part of a trial).
    
    enFr = cs(movieEn+1); % 1st frame of movie 4 in the concatenated movie.
    ff = cs_frtrs_all - enFr;
    trMovEnded = find(ff >= 0, 1) - 1; % in what trial movie 4 ended.
    frs_trEndInMov = enFr - cs_frtrs_all(trMovEnded); % num frames of trial trMovEnded that were recorded in movieEn. (it is perhaps the 1st part of a trial).
    
    % # frames of trials recorded between movieSt and movieEn. The 1st element
    % could be num frames corresponding to part of a trial (bc movieSt happened
    % at the middle of a trial). The last element could be frames recorded
    % without a trial.
    nfrs_perTr_inMovieStEn = [frs_trFirstInMov, framesPerTrialNoNaN(trMovStarted+1 : trMovEnded-1), frs_trEndInMov];
    cs_frtrs = [0 cumsum(nfrs_perTr_inMovieStEn)];

    firstFullTrial = trMovStarted+1; 
    lastFullTrial = trMovEnded-1;

end




%%
%{
save('datasetTempUpdate', 'A', 'C', 'S', 'C_df', 'S_df', 'Df', 'b', 'f', 'srt', 'Ain', 'options', 'P', 'merging_vars')

save('datasetNoTempUpdate', 'A', 'C', 'S', 'C_df', 'S_df', 'Df', 'b', 'f', 'srt', 'Ain', 'options', 'P', 'merging_vars')

%%
Ac = A;
Cc = C;
Sc = S;
C_dfc = C_df;
S_dfc = S_df;
Dfc = Df;
bc = b;
fc = f;
srtc = srt;
Ainc = Ain;
optionsc = options;
Pc = P;
merging_varsc = merging_vars;


%%
figure;
hold on;
plot(mean(Cc,1)/max(mean(Cc,1)))
plot(mean(Sc,1)/max(mean(Sc,1)))
x = [cs_frtrs', cs_frtrs']'+1;
plot(x, [0 1], 'k')
% figure; plot(x, repmat([0 1]', [1, size(x,2)]))
%}


