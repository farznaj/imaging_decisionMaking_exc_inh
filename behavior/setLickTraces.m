function [traces_lick_time, traces_lick_frame] = setLickTraces(alldata, outcomes)
%
% OUTPUT: Cell array of size numTrials x 1. Each cell is a vector of licks
% (0:no lick, 1:center, 2:left, 3:right licks) for a given trial.
%
% traces_lick_time: each sample is 1ms. Sample 1 is the start time of
% trials in bcontrol (ie state0). length(traces_lick_time{tr})= length of
% trial in bcontrol.
%
% traces_lick_frame: each sample is 1 frame. only licks after scopeTTL (ie
% start of imaging) are shown. Sample 1 is the 1st frame of imaging.
% length(traces_lick_frame{tr})=length of framesPerTrial (ie length of
% imaging frames).
%
% Remember you have not applied trs2rmv to the output traces here.


%% Set the time trace for licks (in ms)

% since you are defining time of licks by taking ceil, that means a lick in
% index i of treaces_lick_time, has happened during window [i-1  i] ms.

traces_lick_time = cell(1, length(alldata));
for tr = 1:length(alldata)
%     tr
    % time of center licks relative to the start of the trial in bcontrol (ie state0)
    centLicksRel2TrSt = (alldata(tr).parsedEvents.pokes.C(:,1) * 1000) - (alldata(tr).parsedEvents.states.state_0(2) * 1000);
    centLicksRel2TrSt(isnan(centLicksRel2TrSt)) = []; % not sure why it is occasionally nan. (% why for tr 19 it was nan?!)
    if sum(~centLicksRel2TrSt)
        fprintf('%i center lick(s) found at time 0\n', sum(~centLicksRel2TrSt)) % not sure why it would be 0!    
    end
    centLicksRel2TrSt(~centLicksRel2TrSt) = []; % remove them bc they cause problem for indexing... 
    centLicksRel2TrSt = ceil(centLicksRel2TrSt); 
    
    % time of left licks relative to the start of the trial in bcontrol (ie state0)
    leftLicksRel2TrSt = (alldata(tr).parsedEvents.pokes.L(:,1) * 1000) - (alldata(tr).parsedEvents.states.state_0(2) * 1000);
    leftLicksRel2TrSt = ceil(leftLicksRel2TrSt);
    leftLicksRel2TrSt(isnan(leftLicksRel2TrSt)) = []; % not sure why it is occasionally nan. (% why for tr 19 it was nan?!)
    
    % time of right licks relative to the start of the trial in bcontrol (ie state0)
    rightLicksRel2TrSt = (alldata(tr).parsedEvents.pokes.R(:,1) * 1000) - (alldata(tr).parsedEvents.states.state_0(2) * 1000);
    rightLicksRel2TrSt = ceil(rightLicksRel2TrSt);
    rightLicksRel2TrSt(isnan(rightLicksRel2TrSt)) = []; % not sure why it is occasionally nan. (% why for tr 19 it was nan?!)
    
    % trial duration in ms
    trDur = round(1000 * diff([alldata(tr).parsedEvents.states.state_0(2), alldata(tr).parsedEvents.states.stop_rotary_scope(1)]));


    % form a trace that has the length of trial (trDur) and is all 0s
    % except for when licks happen: center licks indicated by 1, left licks
    % indicated by 2, right licks indicated by 3.
    traces_lick_time{tr} = zeros(trDur, 1);
    traces_lick_time{tr}(centLicksRel2TrSt) = 1;
    traces_lick_time{tr}(leftLicksRel2TrSt) = 2;
    traces_lick_time{tr}(rightLicksRel2TrSt) = 3;
end


% NOTE: Don't be surprised if trial length is different between
% traces_lick_time and traces_lick_frame. In traces_lick_time you compute
% trial length from state_o to the beginning of stop_rotary_scope. In
% traces_lick_frame you compute trials length until the the end of
% stop_rotary_scope (because you get it from nFrames). I think licks don't
% get registered in bctonrol right after starting stop_rotary_scope (ie
% licks during stopScopeDur don't get recorded in bcontrol).


%% Set the frame trace for licks (ie show licks in each frame)... this will be helpful when aligning licks with imaging data.
% Only considering licks that happen after scopeTTL (ie after the start of imaging)
 

% you are not applying trs2rmv here... later in the traces set those trials to nan.
scopeTTLOrigTime = 1;
[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, [], scopeTTLOrigTime, [], outcomes);
% [timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
%     time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
%     setEventTimesRelBcontrolScopeTTL(alldata, [], scopeTTLOrigTime);

traceTimeVec = {alldata.frameTimes};

% center licks
eventTime = centerLicks;
eventTime = cellfun(@(x)x(x>0), eventTime, 'uniformoutput', 0); % only look at centerLicks that happen after scopeTTL.

centLicksRel2ScopeTTLFrame = cell(size(eventTime)); % frame of center licks % eventInds_l_f 
% centLicksRel2ScopeTTLFrame(:) = {NaN};
for tr = 1:length(alldata) % find the frame during which licks happened for each trial.
    if ~isempty(eventTime{tr})
        a = bsxfun(@minus, eventTime{tr}, traceTimeVec{tr}); % numLicks x numFrs
        [~, centLicksRel2ScopeTTLFrame{tr}] = min(abs(a), [], 2); % numLicks x 1 % centLicksRel2ScopeTTL_inFr{tr}: frames (relative to scopeTTL) in which licks happened.
    end
end


% left licks
eventTime = leftLicks;
eventTime = cellfun(@(x)x(x>0), eventTime, 'uniformoutput', 0);

leftLicksRel2ScopeTTLFrame = cell(size(eventTime)); % frame of left licks % eventInds_l_f 
% leftLicksRel2ScopeTTLFrame(:) = {NaN};
for tr = 1:length(alldata) % find the frame during which licks happened for each trial.
    if ~isempty(eventTime{tr})
        a = bsxfun(@minus, eventTime{tr}, traceTimeVec{tr}); % numLicks x numFrs
        [~, leftLicksRel2ScopeTTLFrame{tr}] = min(abs(a), [], 2); % numLicks x 1 % leftLicksRel2ScopeTTL_inFr{tr}: frames (relative to scopeTTL) in which licks happened.
    end
end


% right licks
eventTime = rightLicks;
eventTime = cellfun(@(x)x(x>0), eventTime, 'uniformoutput', 0);

rightLicksRel2ScopeTTLFrame = cell(size(eventTime)); % frame of right licks % eventInds_l_f 
% rightLicksRel2ScopeTTLFrame(:) = {NaN};
for tr = 1:length(alldata) % find the frame during which licks happened for each trial.
    if ~isempty(eventTime{tr})
        a = bsxfun(@minus, eventTime{tr}, traceTimeVec{tr}); % numLicks x numFrs
        [~, rightLicksRel2ScopeTTLFrame{tr}] = min(abs(a), [], 2); % numLicks x 1 % rightLicksRel2ScopeTTL_inFr{tr}: frames (relative to scopeTTL) in which licks happened.
    end
end



traces_lick_frame = cell(1, length(eventTime));
for tr = 1:length(alldata)
    traces_lick_frame{tr} = zeros(alldata(tr).nFrames, 1);
    traces_lick_frame{tr}(centLicksRel2ScopeTTLFrame{tr}) = 1;
    traces_lick_frame{tr}(leftLicksRel2ScopeTTLFrame{tr}) = 2;
    traces_lick_frame{tr}(rightLicksRel2ScopeTTLFrame{tr}) = 3;
end
    




