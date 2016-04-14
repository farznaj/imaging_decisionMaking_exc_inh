function activity = konnerthDeltaFOverF(activity, pmtOffFrs, smoothPts, minPts)
% activity = konnerthDeltaFOverF(activity [, smoothPts] [, minPts])
%
% Take raw fluorescence traces and turn them into delta F / F. Method from
% Jia, Rochefort, Chen, Konnerth (2011) Nature Protocols, but not using
% final exponential filter step from the paper.
%
% Briefly, the method smooths lightly using a boxcar filter, then gets the
% denominator F by taking a running min. This F is subtracted from every
% corresponding point, then we divide by the same F.
%
% activity  -- should be nFrames x nUnits
% smoothPts -- number of points to run box filter over. Default 6.
% minPts    -- number of points to take the running min over.


%% Optional arguments

if ~exist('smoothPts', 'var')
  % 6 is 194 ms at 30.9 frames/s
  smoothPts = 6;
end

if ~exist('minPts', 'var')
  % 93 is 3010 ms at 30.9 frames/s
  minPts = 93;
end


%% Pad front of traces with repeated data
% This is so that smoothing and running min have a reasonable number of
% points to work with at the beginning. We don't want to use a circular pad
% (with data from the end) because bleaching will make late data
% systematically different. So we just repeat the front of the data.

padHeight = minPts - 1;

%{
% note about pmtOffFrames: there are alternative ways to deal with them, eg
% computing runningF for each valid part of the activity trace separately, and then
% reconstructing the final activity trace by inserting NaNs for the pmtOffFrs.
% also remember the codes below need work if more than one secion of the
% activity trace includes pmtOffFrs. (FN)
if any(pmtOffFrs)
    noNaNAct = activity;
    noNaNAct(pmtOffFrs,:) = []; 
    activity = [noNaNAct(1:padHeight, :); activity]; % make sure the padded section doesn't have NaNs, otherwise in imerode it will cause issues.
    clear noNaNAct
else
    activity = [activity(1:padHeight, :); activity];
end
%}

if any(pmtOffFrs)
    d = diff(pmtOffFrs); 
    begs = [1; find(d==-1)+1];  % begining index of valid parts
    ends = [find(d==1) ;length(pmtOffFrs)]; % ending index of valid parts
end


minPtsOrig = minPts;
a0 = activity;
activityFinal = [];

%%
for ivalid = 1:length(begs)
    activity = a0(begs(ivalid):ends(ivalid),:);
    minPts = min(minPtsOrig, floor(size(activity,1)/2));
    padHeight = minPts - 1;
    activity = [activity(1:padHeight, :); activity];
    
%% Smooth
causal = 0; % 0 indicates smoothing window will be centered on the points to be smoothed. 1 indicates smoothing window will be placed on the edge
% activity(isnan(activity)) = 0; % to avoid problems with cumsum.
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


%% Take care of pmtOffFrs.
%{
if any(pmtOffFrs)
    
    pmtfrs = logical([zeros(minPts-1, 1); pmtOffFrs]);
    pmtOffEnd = find(pmtfrs,1,'last');
    if ~causal
        pmtfrs(pmtOffEnd+1: pmtOffEnd+smoothPts/2) = true; % to correct for the few data points after pmtOffFrs in runningF that have bad values due to smoothing in boxFilter over smoothPts.
    else
        pmtfrs(pmtOffEnd+1: pmtOffEnd+smoothPts) = true;
    end
    
    % correct for bad values at the edges of runningF:

    % imerode sets to 0 the values of frames at the ending edge of pmtfrs, so we replace
    % them with the first legitimate values in runningF that happen after
    % pmtfrs; so runningF for frs2replace will not be exact, but good
    % enough approximation.
    if ~causal
        frs2replace = pmtOffEnd+1 : pmtOffEnd + minPts + smoothPts/2; 
    else
        frs2replace = pmtOffEnd+1 : pmtOffEnd + minPts + smoothPts; 
    end    
    runningF(frs2replace,:) = runningF(frs2replace(end):frs2replace(end)+length(frs2replace)-1,:); 
    
    % if filtering was acausal, we need to correct for the values of few
    % points immediately before pmtfrs as well. Again we replace them with
    % the first legitimate values in runningF that happen before pmtfrs.
    if ~causal
        pmtOffBeg = find(pmtfrs,1,'first');
        frs2replace = pmtOffBeg-smoothPts/2-1 : pmtOffBeg-1; 
        runningF(frs2replace,:) = runningF(frs2replace(1)-length(frs2replace)+1:frs2replace(1),:);    
    end
    
    % Set pmtOffFrames in runningF and activity to NaN
    runningF(pmtfrs,:) = NaN;
    activity(pmtfrs,:) = NaN;
    
end
%}

%% Compute deltaF/F

activity = (activity - runningF) ./ runningF;


%% Remove pad

activity = activity(padHeight+1:end, :);

% add back NaNs
if ivalid<length(begs)
    activity = [activity; NaN(length(ends(ivalid)+1 : begs(ivalid+1)-1), size(activity,2))];
end

activityFinal = [activityFinal; activity];

end




