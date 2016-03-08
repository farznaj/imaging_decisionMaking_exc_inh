% 455mm is the periphary of the wheel.

% compare the length of the trial (start to stop of scan and rotary)
% between bcontrol, mscan, and rotary.
diffBcont_mscan_rot = NaN(length(alldata), 2);
diffRot_mscan = NaN(length(alldata), 2);
for itr = 1:length(alldata) % if alldata is not cleaned, you will need to get rid of its last trial.
    if isfield(alldata(itr).parsedEvents.states, 'trial_start_rot_scope')
        startOnOff = alldata(itr).parsedEvents.states.trial_start_rot_scope;
    else
        startOnOff = alldata(itr).parsedEvents.states.start_rotary_scope(1, :); % 1st state that sent the scope ttl.
    end
    
    len_bcontrol = (alldata(itr).parsedEvents.states.stop_rotary_scope(1)*1000 + .5*1000) - (startOnOff(1)*1000); % ms, length of start to stop of scan (and rotary) in bcontrol (when scanning/rotary was started untiil when they were stopped. rotary may be different if wheelPostTrialToRecord was not 0)
    len_mscan = framesPerTrial(itr)*frameLength; % ms, length of the trial in mscan
    len_rotary = length(alldata(itr).wheelRev)*alldata(itr).wheelSampleInt - alldata(itr).wheelPostTrialToRecord; % ms, length of the rotary signal excluding wheelPostTrialToRecord.
    diffBcont_mscan_rot(itr,:) = [len_bcontrol-len_mscan  len_bcontrol-len_rotary];
    diffRot_mscan(itr,:) = len_rotary - len_mscan;
end
% remember the max rotary array is defined in arduino 25000points (ms), so
% if you see a big value for len_bcontrol-len_rotary, it is perhaps bc of
% that.
figure; plot(diffBcont_mscan_rot)
figure; plot(diffRot_mscan)


%% plot the wheelRev signal for trials one by one
wheelTimeRes = alldata(1).wheelSampleInt;
figure;
for itr = 1:length(alldata) % 102
    wheelTimes = wheelTimeRes / 2 +  wheelTimeRes * (0:length(alldata(itr).wheelRev)-1);
    wheelRev = alldata(itr).wheelRev;
    wheelRev = wheelRev - wheelRev(1); % since the absolute values don't matter, I assume the position of the wheel at the start of a trial was 0. I think values are negative, bc when the mouse moves forward, rotary turns counter clock wise.
    
    plot(wheelTimes, wheelRev)
    %     plot(wheelTimes(1:end-1), diff(wheelRev)) % speed of wheel
    xlabel('Time from the beginning of start\_rotary\_scope (ms) ',  'Interpreter', 'tex')
    ylabel('Wheel revolution')
    
    pause,
end


%% concatenate wheelRev for all trials and plot it.
% values seem to range from -16 to 16, not sure why, but what matters is
% the change not the absolute values.
a = {alldata.wheelRev};
figure; plot(cell2mat(a'))
