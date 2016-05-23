% Align licks on trial events and look at the average across trials.

% Choose below what licks you want to analyze.
lickInds = [1,2,3]; % all licks
% lickInds = [1]; % center lick
% lickInds = [2]; % left lick
% lickInds = [3]; % right lick


%% Set the traces for licks (in time and frame resolution)

[traces_lick_time, traces_lick_frame] = setLickTraces(alldata);
% Remember you have not applied trs2rmv to the output traces.


%%
evT = {'timeStop'}; % {'1', 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
% 'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',... 
% 'timeReward', 'timeCommitIncorrResp'}; %,... 


%% Plot licks in ms resolution. Show all licks during a trial. Align licks on different trial events.
% this would be useful if you don't want to align with imaging data.

% Get the time of events in ms relative to start of each trial in bcontrol
scopeTTLOrigTime = 0;
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime);


%%%%%
f = figure;
% doplots = 0;

for i = 1:length(evT)
    
    eventTime = eval(evT{i});     % eventTime = timeInitTone;
    
    if length(eventTime)==1 && eventTime==1
        % align on the first frame (when scanning started).
        eventTime = ones(size(alldata));
        eventTime(trs2rmv) = NaN;
        
    elseif iscell(eventTime)
        eventTime = cellfun(@(x)x(1), eventTime);
    end
    
%     eventTime = round(eventTime);
    
    
    %% Align lick trace on eventTime
    
    traces = traces_lick_time;
    traces = cellfun(@(x)ismember(x, lickInds), traces, 'uniformoutput', 0); % assign all licks to 1. don't distinguish among licks.
    
    % [traceEventAlign_licks, timeEventAlign_licks] = triggerAlignTraces(traces, eventTime);
    [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces, eventTime);
    
    
    %% Plot
    
    figure(f)
    subplot(2,length(evT),length(evT)*0+i), hold on
    top = nanmean(nanmean(traceEventAlign,3),2); % average across trials and neurons.
    plot(top)
    %     plot(timeEventAlign, top)
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'last');
    
    %     plot([0 0],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    e = find(timeEventAlign >= 0, 1);
    plot([e e], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    
%     xlim([xl1 xl2])
    xlim([e-1000 e+1000])
    %     xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
    xlabel('Time (ms)')
    ylabel('Average licks')
    title(evT{i})
    
    
    %{
    % average licks for each frame
    l = length(timeEventAlign);
    t = unique([round(e : frameLength : l), round(e : -frameLength : 1)]); % e + frameLength-1

    traceEventAlignAveFr = NaN(length(t)-1, size(traceEventAlign,2), size(traceEventAlign,3));
    for i = 1:length(t)-1 % 1011; % [m,i]=(min(abs(t-e)))
        traceEventAlignAveFr(i,:,:) = nanmean(traceEventAlign(t(i):t(i+1)-1,:,:), 1);
    end

    figure;
    hold on
    top = nanmean(nanmean(traceEventAlignAveFr,3),2);
    plot(top)
    xlim([xl1 xl2]/32)
    %}
    
end





%% Plot licks in frame resolution. Show only licks after scopeTTL. Align licks on different trial events.
% this would be useful if you are comparing with imaging data.

% Get the time of events in frame relative to start of scopeTTL.

scopeTTLOrigTime = 1;
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime);


% f = figure;
doplots = 0;

for i = 1:length(evT)
    
    eventTime = eval(evT{i});
    traces = traces_lick_frame; % alldataSpikesGood; %  traces to be aligned.
    traces = cellfun(@(x)ismember(x, lickInds), traces, 'uniformoutput', 0); % assign all licks to 1. don't distinguish among licks.
    
    [traceEventAlign, timeEventAlign, nvalidtrs, traceEventAlign_wheelRev, ...
        timeEventAlign_wheelRev, nvalidtrs_wheel] = ...
        avetrialAlign_noTrGroup(eventTime, traces, alldata, frameLength, trs2rmv, doplots);
    
    
    %% plot licks
    
    figure(f)
    subplot(2,length(evT),length(evT)*1+i), hold on
    % subplot(223), hold on
    top = nanmean(nanmean(traceEventAlign,3),2); % average across trials and neurons.
    plot(top)
    %     plot(timeEventAlign, top)
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'last');
    
    %     plot([0 0],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    e = find(timeEventAlign > 0, 1);
    plot([e e], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    
    xlim([xl1 xl2])
    %     xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
    xlabel('Frame')
    ylabel('Average licks')
    title(evT{i})
    
    
end


%%

%{
% the following 2 should be close. Trialcode duration : the given values vs the value computed from bcontrol states
[alldata(itr).durTrialCode, ...
diff([alldata(itr).parsedEvents.states.trialcode1(1) alldata(itr).parsedEvents.states.wait_for_initiation(1)])]
%}


