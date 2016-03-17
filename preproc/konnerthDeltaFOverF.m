function dFOF = konnerthDeltaFOverF(activity, pmtOffFrs, smoothPts, minPts)
% activity = konnerthDeltaFOverF(activity [, pmtOffFrs] [, smoothPts] [, minPts])
%
% Take raw fluorescence traces and turn them into delta F / F. Method from
% Jia, Rochefort, Chen, Konnerth (2011) Nature Protocols, but not using
% final exponential filter step from the paper.
%
% Briefly, the method smooths lightly using a boxcar filter, then gets the
% denominator F by taking a running min. This F is subtracted from every
% corresponding point, then we divide by the same F.
%
% activity  -- nFrames x nUnits
% pmtOffFrs -- vector of length nFrames: which frames to ignore. Empty
%              input is the same as all false.
% smoothPts -- number of points to run box filter over. Default 6 (194 ms)
% minPts    -- number of points to take the running min over. Default 93
%              (~3 seconds)


%% Optional arguments

if ~exist('smoothPts', 'var')
    % 6 is 194 ms at 30.9 frames/s
    smoothPts = 6;
end

if ~exist('minPts', 'var')
    % 93 is 3010 ms at 30.9 frames/s
    minPts = 93;
end


%% 1st dimension of activity should be frames. Make sure this is the case.

if ~exist('pmtOffFrs', 'var') || isempty(pmtOffFrs)
  pmtOffFrs = false(1, size(activity, 1));
  dimsOk = 1;
  
else
  dimsOk = 1;
  [d1, d2] = size(activity);
  if d1 ~= length(pmtOffFrs)
    if d2 ~= length(pmtOffFrs)
      error('konnerthDeltaFOverF: length of pmtOffFrs does not match either dimension of activity');
    end
    
    activity = activity';
    dimsOk = 0;
    % set back dFOF dimensions to the original input activity
  end
end


%% Set some parameters

% 0 indicates smoothing window will be centered on the points to be
% smoothed. 1 indicates smoothing window will only include points before
% this point (causal influence).
causal = 0;

if any(pmtOffFrs)
    d = diff(pmtOffFrs);
    begs = [1; find(d==-1)+1];  % begining index of valid parts
    ends = [find(d==1); length(pmtOffFrs)]; % ending index of valid parts
else
    begs = 1;
    ends = length(pmtOffFrs);
end

minPtsOrig = minPts;
activityOrig = activity;
dFOF = [];


%% Loop over valid parts of the activity trace (i.e., parts without pmtOffFrames)

for ivalid = 1:length(begs)
    
    activity = activityOrig(begs(ivalid):ends(ivalid),:);
    minPts = min(minPtsOrig, size(activity,1));
    
    % Pad front of traces with repeated data
    % This is so that smoothing and running min have a reasonable number of
    % points to work with at the beginning. We don't want to use a circular pad
    % (with data from the end) because bleaching will make late data
    % systematically different. So we just repeat the front of the data.
    padHeight = minPts - 1;
    activity = [activity(1:padHeight, :); activity];
    
    %% Smooth    
    runningF = boxFilter(activity, smoothPts, 1, causal);
    
    
    %% Running min
    % We can perform a running min using an operation normally performed on
    % images called "erosion". Erosion replaces each element of an image with a
    % min over a neighborhood, defined by a "structuring element". By making
    % this structuring element the right shape, we can get a running min in one
    % dimension.
    
    % The structuring element is centered on the point of interest, so we want
    % to have 1's for the first half (including the center element), and 0's
    % for the "future" elements
    structElem = [ones(minPts, 1); zeros(minPts-1, 1)];
    runningF = imerode(runningF, structElem);
    
    
    %% Compute deltaF/F    
    activity = (activity - runningF) ./ runningF;
    
    
    %% Remove pad    
    activity = activity(padHeight+1:end, :);
    
    
    %% Add NaNs corresponding to pmtOffFrames and recostruct the final dFOF trace
    
    if ivalid < length(begs)
        activity = [activity; NaN(length(ends(ivalid)+1 : begs(ivalid+1)-1), size(activity,2))];
    end
    
    dFOF = [dFOF; activity];
    
end



%% Set back dFOF dimensions to the dimensions of the original input activity

if ~dimsOk
    dFOF = dFOF';
end


