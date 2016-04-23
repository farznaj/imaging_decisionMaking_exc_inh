function [alldata, mscanLag] = mergeActivityIntoAlldata_fn(alldata, activity, framesPerTrial, ...
    trialNumbers, frame1RelToStartOff, badFrames, pmtOffFrames, minPts, dFOF, S_df)
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
%  badFrames     -- logical array indicating whether each frame was bad (logical vector, indicating whether each frame was
%                   motion corrected by more than maxMaskWidth)
%  frameTimes    -- time that each frame started relative to when bcontrol signal was sent. (matt: the time in the trial that each frame started)
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

if nansum(framesPerTrial) ~= nFrames % FN: this is fine because if recording was stopped during wait4init, frameCount text file will not show the frames of the last trial (during which mscan was stopped) but the activity trace will have those frames.
    %   error('framesPerTrial and activity do not contain same number of frames!');
    fprintf('%d frames were recorded in the last trial during which mscan was stopped.\n', nFrames - nansum(framesPerTrial))
end


%% Prepare new alldata fields

[alldata.hasActivity] = deal(false);
[alldata.nFrames] = deal(0);
[alldata.anyBadFrames] = deal(NaN);


%% Prepare deltaF/F
if ~exist('dFOF', 'var')
    smoothPts = 6;
    dFOF = konnerthDeltaFOverF(activity, pmtOffFrames, smoothPts, minPts);
end

%% Distribute activity to trials

adTr = 1;   % Index in alldata that we're working on
adTrNums = [alldata.trialId];    % trialId's from alldata
activityInd = 1;
nMissing = 0;
mscanLag = NaN(1, length(trialNumbers)); % duration (ms) that MScan lagged behind in executing bcontrol command signal % To get the accurate time points of each frame add this to frameLength/2 + frameLength*(0:framesPerTrial(imTr)-1);

frameLength = 1000 / frameRate;


csfrs = [0 cumsum(framesPerTrial)];
f = find((csfrs-size(activity,1))>0, 1)-2;
if isempty(f)
    lastImTr = length(trialNumbers);
else
    warning('FN need to check this. This shouldnt happen if you are using activity trace that corresponds to entire session')
    lastImTr = f; % if the activity trace doesn't correspond to the entire movie.
end

for imTr = 1:lastImTr
    % Grab the trial number that the imaging data thinks this is. Note that
    % this value may indicate that we don't know the trial number for this
    % set of frames
    imTrNum = trialNumbers(imTr);
    
    % If the trial number is known for this set of frames, find the
    % corresponding trial in alldata and merge in
    if imTrNum > 0
        while adTr <= length(adTrNums) && adTrNums(adTr) < imTrNum
            adTr = adTr + 1; % Farz: adTr keeps going up until it matches imTrNum. This is bc imaging trials (trialNumbers) might miss some of the bcontrol trials.
        end
        
        % Double-check that we have a match
        if adTr <= length(adTrNums) && adTrNums(adTr) == imTrNum && ~isnan(framesPerTrial(imTr)) % Farz added the isnan term, bc when you stop the mouse during wait4init state, trialStart and trialCode get recorded, however no frameCounts will be recorde in the text file, which will turn to NaN in framesPerTrialStopStart2An_fn.
            alldata(adTr).activity = activity(activityInd:activityInd+framesPerTrial(imTr)-1, :); % frames x units
            alldata(adTr).dFOF = dFOF(activityInd:activityInd+framesPerTrial(imTr)-1, :); % frames x units
            if exist('S_df', 'var')
                alldata(adTr).spikes = S_df(activityInd:activityInd+framesPerTrial(imTr)-1, :); % frames x units
            end
            alldata(adTr).hasActivity = true;
            alldata(adTr).nFrames = framesPerTrial(imTr);
            alldata(adTr).badFrames = badFrames(activityInd:activityInd+framesPerTrial(imTr)-1); % frames x 1
            alldata(adTr).anyBadFrames = any(alldata(adTr).badFrames);
            alldata(adTr).pmtOffFrames = pmtOffFrames(activityInd:activityInd+framesPerTrial(imTr)-1); % frames x 1
            alldata(adTr).anyPmtOffFrames = any(alldata(adTr).pmtOffFrames);
            
            % remember that for badAlignTrStartCode, trialStartMissing you are
            % still setting alldata.frameTimes but you need to take that into
            % account when analyzing the data!
            
            % Figure out time alignment
            % This is the first state, so every trial has these values
            %       state0 = alldata(adTr).parsedEvents.states.state_0(2); % Farz commented. Not needed.
            %       startOnOff = alldata(adTr).parsedEvents.states.wait_in_center(1, :); % Farz commented. Not applicable.
            %       firstFrameStart = 1000 * (startOnOff(2) - state0) + frame1RelToStartOff(imTr); % Farz commented. Not applicable.
            
            if isfield(alldata(adTr).parsedEvents.states, 'trial_start_rot_scope')
                startOnOff = alldata(adTr).parsedEvents.states.trial_start_rot_scope;
            else
                startOnOff = alldata(adTr).parsedEvents.states.start_rotary_scope(1, :); % 1st state that sent the scope ttl.
            end
            firstFrameStart = 1000 * diff(startOnOff) + frame1RelToStartOff(imTr); % duration (ms) that MScan lagged behind in executing bcontrol command signal % remember bcontrol states are in sec. % frame1RelToStartOff shows the lag between when bcontrol sent scopeTTL (+ trialStart) and when MScan started scanning. If it is -499 (ie the duration of start_rotary_scope) there was no lag. If it is >-499, it means MScan did not save the entire duration of start_rotary_scope, hence imaging started at a delay. If it <-499, mouse has entered state start_rotary_scope2.
            mscanLag(adTr) = firstFrameStart;
            % alldata.frameTimes is relative to when bcontrol sends the scope ttl.
            alldata(adTr).frameTimes = firstFrameStart + frameLength / 2 + ...
                frameLength * (0:framesPerTrial(imTr)-1); % frameLength / 2 is added to set frameTimes to the center of a frame (instead of the beginning).
            
            
            %{
            % wheelRev times: FN added and commented this part out bc Arduino's serial part doesn't have the same delay as MScan in writing data. If you want to measure it read Arduino's signal every ms, and send a fixed say 10ms signal from bcontrol to Arduino and see how long of it gets writtten to the Arduino's serial port.             
            if isfield(alldata(adTr), 'wheelRev')
                wheelTimeRes = alldata(adTr).wheelSampleInt;
                alldata(adTr).wheelTimes = firstFrameStart + wheelTimeRes / 2 +  wheelTimeRes * (0:length(alldata(adTr).wheelRev)-1);
            end
            %}
            
        else
            
            if ~isnan(framesPerTrial(imTr)) % Farz 
                if imTrNum == adTrNums(end) + 1 % || imTrNum == adTrNums(end) % second term added by Farz.
                    fprintf('Missing the last trial of alldata, as is typical.\n');
                else
                    warning([num2str(imTrNum),'Missing a middle trial in alldata that is present in imaging data. This should never happen!']);
                end
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
