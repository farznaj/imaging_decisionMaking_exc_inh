function lickAlign(lickInds, evT, outcome2ana, stimrate2ana, strength2ana, trs2rmv, outcomes, stimrate, cb, alldata, frameLength)
% Align licks on trial events and look at their average across trials.
% Remember these are lick traces (not ca imaging traces).
% Example input variables:
%{
lickInds = [1, 2,3]; % Licks to analyze % 1: center lick, 2: left lick; 3: right lick
outcome2ana = 'all'; % 'all'; 1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
stimrate2ana = 'all'; % 'all'; 'HR'; 'LR';
strength2ana = 'all'; % 'all'; 'eary'; 'medium'; 'hard';

% evT = {'timeStop'}; % what events to align trials on and plot?
evT = {'1', 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
    'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
    'timeReward', 'timeCommitIncorrResp', 'timeStop'};

%}


thStimStrength = 3; % 2; % threshold of stim strength for defining hard, medium and easy trials.


%%
s = (stimrate-cb)'; 
allStrn = unique(abs(s));
switch strength2ana
    case 'easy'
        str2ana = (abs(s) >= (max(allStrn) - thStimStrength));
    case 'hard'
        str2ana = (abs(s) <= thStimStrength);
    case 'medium'
        str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(allStrn) - thStimStrength))); % intermediate strength
    otherwise
        str2ana = true(1, length(outcomes));
end

if strcmp(outcome2ana, 'all')
    os = sprintf('%s', outcome2ana);
    outcome2ana = -5:1;    
else
    os = sprintf('%i', outcome2ana);
end

switch stimrate2ana
    case 'HR'
        sr2ana = s > 0;
    case 'LR'
        sr2ana = s < 0;
    otherwise
        sr2ana = true(1, length(outcomes));
end

trs2ana = (ismember(outcomes, outcome2ana)) & str2ana & sr2ana;
% trs2ana = outcomes==0 & timeLastSideLickToStopT > 1000;
% trs2ana(163:end) = false; % first trials 
% trs2ana(1:162) = false; % last trials

sl = repmat('%d ', 1, length(lickInds));
topn = sprintf(['licks ', sl, ', %s outcomes, %s strengths, %s stimulus: %i trials'], lickInds, os, strength2ana, stimrate2ana, sum(trs2ana));
disp(['Analyzing ', topn])


%% Set the traces for licks (in time and frame resolution)

[traces_lick_time, traces_lick_frame] = setLickTraces(alldata, outcomes);
% Remember you have not applied trs2rmv to the output traces.


%%
alldatanow = alldata(trs2ana);
trs2rmvnow = find(ismember(find(trs2ana), trs2rmv));


%% Plot licks in ms resolution. Show all licks during a trial. Align licks on different trial events.
% this is useful if you don't want to align with imaging data.

% Get the time of events in ms relative to the start of each trial in bcontrol
scopeTTLOrigTime = 0;
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime, [], outcomes);


f = figure('name', topn, 'position', [520         284        1401         459]);
% doplots = 0;

for i = 1:length(evT)
    
    disp(['------- ', evT{i}, ' -------'])
    
    eventTime = eval(evT{i});     % eventTime = timeInitTone;
    
    if length(eventTime)==1 && eventTime==1
        % align on the first frame (when scanning started).
        eventTime = ones(size(alldata));
        eventTime(trs2rmvnow) = NaN;
        
    elseif iscell(eventTime)
        eventTime = cellfun(@(x)x(1), eventTime);
    end
    eventTime = eventTime(trs2ana);
    %     eventTime = round(eventTime);
    if all(isnan(eventTime))
        warning('Choose outcome2ana and evT wisely! eg. you cannot choose outcome2ana=1 and evT="timeCommitIncorrResp"')
        error('Something wrong; eventTime is all NaNs. Could be due to improper choice of outcome2ana and evT{i}!')
    end
    
    
    %% Align lick trace on eventTime
    
    traces = traces_lick_time;
    traces = cellfun(@(x)ismember(x, lickInds), traces, 'uniformoutput', 0); % Only extract the licks that you want to analyze (center, left or right).
    traces = traces(trs2ana);
    
    % [traceEventAlign_licks, timeEventAlign_licks] = triggerAlignTraces(traces, eventTime);
    [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces, eventTime);
    
    
    %% Plot
    
    figure(f)
    subplot(2,length(evT),length(evT)*0+i), hold on
    
    av = nanmean(traceEventAlign,3); % frames x units. (average across trials).
    top = nanmean(av,2); % average across neurons.
%     tosd = nanstd(av, [], 2);
%     tosd = tosd / sqrt(size(av, 2)); % plot se
    
    e = find(timeEventAlign >= 0, 1);
%     boundedline((1:length(top))-e, top, tosd, 'alpha')
    plot((1:length(top))-e, top)
    %     plot(top)
    %     plot(timeEventAlign, top)
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'last');
    
    xlim([xl1 xl2]-e)
    
    plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    
    %{
    top = nanmean(nanmean(traceEventAlign,3),2); % average across trials and neurons.
    plot(top)
    %     plot(timeEventAlign, top)
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'last');
    
    %     plot([0 0],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    e = find(timeEventAlign >= 0, 1);
    plot([e e], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r-.')
    
    xlim([xl1-100 xl2+100])
    %     xlim([xl1 xl2])
    %     xlim([e-1000 e+1000])
    %     xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
    %}
    if i==1
        xlabel('Time (ms)')
        ylabel('Fraction trials with licks')
    end
    if i>1
        title(evT{i}(5:end))
    else
        title(evT{i})        
    end
    
    
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

    
    %% Look at the raster of licks (eg compare how left and right lick patterns are different
    %{
    a = squeeze(traceEventAlign);
    a = a(e-500:end,:);

    figure; imagesc(a')
    %}

end





%% Plot licks in frame resolution. Show only licks after scopeTTL. Align licks on different trial events.
% this is useful if you are comparing with imaging data.

% Get the time of events in frame relative to start of scopeTTL.

scopeTTLOrigTime = 1;
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime, [], outcomes);


% alldatanow = alldata(trs2ana);
% trs2rmvnow = find(ismember(find(trs2ana), trs2rmv));


% f = figure;
doplots = 0;

for i = 1:length(evT)
    
    eventTime = eval(evT{i});
    if length(eventTime)>1
        eventTime = eventTime(trs2ana);
    end
    
    traces = traces_lick_frame; % alldataSpikesGood; %  traces to be aligned.
    traces = cellfun(@(x)ismember(x, lickInds), traces, 'uniformoutput', 0); % assign all licks to 1. don't distinguish among licks.
    traces = traces(trs2ana);
    
    [traceEventAlign, timeEventAlign, nvalidtrs, traceEventAlign_wheelRev, ...
        timeEventAlign_wheelRev, nvalidtrs_wheel] = ...
        avetrialAlign_noTrGroup(eventTime, traces, alldatanow, frameLength, trs2rmvnow, doplots);
    
    
    %% plot licks
    
    figure(f)
    subplot(2,length(evT),length(evT)*1+i), hold on
    
    av = nanmean(traceEventAlign,3); % frames x units. (average across trials).
    top = nanmean(av,2); % average across neurons.
%     tosd = nanstd(av, [], 2);
%     tosd = tosd / sqrt(size(av, 2)); % plot se
    
    e = find(timeEventAlign >= 0, 1);
%     boundedline((1:length(top))-e, top, tosd, 'alpha')
    plot((1:length(top))-e, top)
    %     plot(top)
    %     plot(timeEventAlign, top)
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'last');
    
    xlim([xl1 xl2]-e)
    
    plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
    
    %{
    % subplot(223), hold on
    top = nanmean(nanmean(traceEventAlign,3),2); % average across trials and neurons.
    plot(top)
    %     plot(timeEventAlign, top)
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'last');
    
    %     plot([0 0],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    e = find(timeEventAlign > 0, 1);
    plot([e e], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r-.')
    
    xlim([xl1 xl2])
    %     xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
    %}
    if i==1
        xlabel('Frame')
        ylabel('Fraction trials with licks')
    end
    %     title(evT{i})
    
    
end


%%

%{
% the following 2 should be close. Trialcode duration : the given values vs the value computed from bcontrol states
[alldata(itr).durTrialCode, ...
diff([alldata(itr).parsedEvents.states.trialcode1(1) alldata(itr).parsedEvents.states.wait_for_initiation(1)])]
%}


