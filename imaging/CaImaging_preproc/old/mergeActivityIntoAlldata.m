function alldata = mergeActivityIntoAlldata(alldata, activity, framesPerTrial, ...
  trialNumbers, frame1RelToStartOff, badFrames)
% alldata = mergeActivityIntoAlldata(alldata, activity, framesPerTrial, ...
%   trialNumbers, frame1RelToStartOff [, badFrames])
%
% INPUTS
% activity should be the nFrames x nUnits activity matrix containing the
%   fluorescence traces from all the trials you have available, from
%   applyFijiROIsToTifs() or another extraction function
% framesPerTrial, trialNumbers, and frame1RelToStartOff should be from
%   framesPerTrialStopStart3An()
% badFrames (optional) should be from preprocessCaMovies()
%
% Adds several fields to your alldata struct:
%  hasActivity   -- whether there was activity for that trial
%  nFrames       -- the number of frames/points for that trial
%  anyBadFrames  -- whether any of the frames for that trial were bad
%  badFrames     -- logical array indicating whether each frame was bad
%  frameTimes    -- the time in the trial that each frame started
%  activity      -- the nFramesOnThisTrial x nUnits matrix of raw
%                   fluorescence activity
%  dFOF          -- the nFramesOnThisTrial x nUnits matrix of delta F / F
%                   computed using konnerthDeltaFOverF()


%% Parameters

frameRate = 30.9;


%% Optional arguments

nFrames = size(activity, 1);

if ~exist('badFrames', 'var') || isempty(badFrames)
  badFrames = false(1, nFrames);
end


%% Error checking

if length(framesPerTrial) ~= length(trialNumbers)
  error('framesPerTrial and trialNumbers have different lengths!');
end

% Farz for now
%{
if sum(framesPerTrial) ~= nFrames
  error('framesPerTrial and activity do not contain same number of frames!');
end
%}


%% Prepare new alldata fields

[alldata.hasActivity] = deal(false);
[alldata.nFrames] = deal(0);
[alldata.anyBadFrames] = deal(NaN);


%% Prepare deltaF/F

dFOF = konnerthDeltaFOverF(activity);


%% Distribute activity to trials

adTr = 1;   % Index in alldata that we're working on
adTrNums = [alldata.trialId];    % trialId's from alldata
activityInd = 1;
nMissing = 0;

frameLength = 1000 / frameRate;

for imTr = 1:length(trialNumbers)
  % Grab the trial number that the imaging data thinks this is. Note that
  % this value may indicate that we don't know the trial number for this
  % set of frames
  imTrNum = trialNumbers(imTr);
  
  % If the trial number is known for this set of frames, find the
  % corresponding trial in alldata and merge in
  if imTrNum > 0
    while adTr <= length(adTrNums) && adTrNums(adTr) < imTrNum
      adTr = adTr + 1;
    end
    
    % Double-check that we have a match
    if adTr <= length(adTrNums) && adTrNums(adTr) == imTrNum
      alldata(adTr).activity = activity(activityInd:activityInd+framesPerTrial(imTr)-1, :);
      alldata(adTr).dFOF = dFOF(activityInd:activityInd+framesPerTrial(imTr)-1, :);
      alldata(adTr).hasActivity = true;
      alldata(adTr).nFrames = framesPerTrial(imTr);
      alldata(adTr).badFrames = badFrames(activityInd:activityInd+framesPerTrial(imTr)-1);
      alldata(adTr).anyBadFrames = any(alldata(adTr).badFrames);
      
      % Figure out time alignment
      % This is the first state, so every trial has these values
      state0 = alldata(adTr).parsedEvents.states.state_0(2);
      startOnOff = alldata(adTr).parsedEvents.states.wait_in_center(1, :);
      firstFrameStart = 1000 * (startOnOff(2) - state0) + frame1RelToStartOff(imTr);
      
      alldata(adTr).frameTimes = firstFrameStart + frameLength / 2 + ...
        frameLength * (0:framesPerTrial(imTr)-1);
    else
      if imTrNum == adTrNums(end) + 1
        fprintf('Missing the last trial of alldata, as is typical.\n');
      else
        warning('Missing a middle trial in alldata that is present in imaging data. This should never happen!');
      end
    end
  else
    nMissing = nMissing + 1;
  end
  
  % No matter what, advance to the next set of frames
  activityInd = activityInd + framesPerTrial(imTr);
end

if nMissing > 0
  fprintf('%d trials could not be matched with imaging data due to alignment failure. This does not affect other trials.\n', nMissing);
end

fprintf('Data merged.\n');
