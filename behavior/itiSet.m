function [iti_noscan, iti_gui_state] = itiSet(alldata)
% ITI in msec.
% computes ITI preceding each trial.
% drop the 1st element of iti_noscan (and iti_gui_state) to get ITI following each trial.


% duration between trials that imaging data is not collected, ie the time
% between stopRotaryScope state of the previous trial and startRotaryScope
% of the current trial.
iti_noscan = zeros(length(alldata),1); % iti_noscan(1)=0 indicates preceding trial 1, iti_noscan is zero (which makes sense.)
for itr = 2:length(alldata)
%     if ~isempty(alldata(itr).parsedEvents) & ~isempty(alldata(itr-1).parsedEvents)
        if isfield(alldata(itr).parsedEvents.states, 'trial_start_rot_scope')
            iti_noscan(itr) = alldata(itr).parsedEvents.states.trial_start_rot_scope(1)*1000 - (alldata(itr-1).parsedEvents.states.stop_rotary_scope(1)+.5)*1000;
        else
            iti_noscan(itr) = alldata(itr).parsedEvents.states.start_rotary_scope(1)*1000 - (alldata(itr-1).parsedEvents.states.stop_rotary_scope(1)+.5)*1000;
        end
%     else
%         iti_noscan(itr) = nan;
%     end
end

% frameLength = 1000/30.9; % msec.
% nFrames_iti = round(iti_noscan/frameLength); % nFrames between trials (during iti) that was not imaged.


% the following 2 values should be identical: 2 ways of computing
% iti value: 1) value set in gui. 2) how long mouse spent in state iti.
% The nice thing is that if computed ITI turns out much larger
% than the set ITI, it indicates trial was paused during preced iti :), for
% this reason mouse spent more time in the iti state than value(ITI).
iti_gui_state = NaN(length(alldata),2);
for itr = 1:length(alldata)
%     if ~isempty(alldata(itr).parsedEvents)
        i1 = diff(alldata(itr).parsedEvents.states.iti(end,:)*1000,[],2);
        if ~isempty(alldata(itr).parsedEvents.states.iti2)
            i2 = diff(alldata(itr).parsedEvents.states.iti2(end,:)*1000,[],2);
        else
            i2 = [];
        end
        iti_computed = max([i1,i2]) + 1000; % 1000ms is added bc duration of states iti and iti2 = value(ITI)-1
        iti_gui_state(itr,:) = [alldata(itr).iti*1000  iti_computed];
%     end
end


%%
% iti_code = zeros(length(alldata),1); % iti_noscan(1)=0 indicates preceding trial 1, iti_noscan is zero (which makes sense.)
% for itr = 2:length(alldata)
%     iti_code(itr) = alldata(itr).parsedEvents.states.trialcode1(1) - alldata(itr-1).parsedEvents.states.trialcode1(1);
% end


