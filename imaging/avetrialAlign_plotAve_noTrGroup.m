function avetrialAlign_plotAve_noTrGroup(evT, outcome2ana, stimrate2ana, strength2ana, trs2rmv, outcomes, stimrate, cb, alldata, alldataDfofGood, alldataSpikesGood, frameLength, timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, timeStop)
% This functions plots average traces across all neurons and all trials
% aligned on particular trial events.
% It calls avetrialAlign_noTrGroup which calls triggerAlignTraces for
% alignment : you don't specify prePost frames, it uses NaNs and takes all
% frames into the average. Then you only look at parts of the trace that
% has enough number of trials contributing to the average.
% 
% To align on licks ('centerLicks', 'leftLicks', 'rightLicks), use script 
% avetrialAlign_plotAve_noTrGroup_licks
%
% Example input variables:
%{
outcome2ana = 1; % 1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation, -4: no center commit, -5: no side commit
stimrate2ana = 'HR'; % 'all'; 'HR'; 'LR';
strength2ana = 'all'; % 'all'; 'easy'; 'medium'; 'hard'; 
evT = {'timeStop'}; % what events to align trials on and plot?

%}

% evT = {'1', 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
%     'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
%     'timeReward', 'timeCommitIncorrResp', 'timeStop'};

% 'centerLicks', 'leftLicks', 'rightLicks'};
% time1stCenterLick, time1stCorrectResponse, timeStop
% eventTime = cellfun(@(x)x(1),timeInitTone); %timeInitTone; % first trial initi tone.

thStimStrength = 3; % 2; % threshold of stim strength for defining hard, medium and easy trials.
unitFrame = 1; % if 1, x axis will be in units of frames. If 0, x axis will be in units of time.


%% Set initial vars.

% remember 1stIncorrTry is not necessarily outcomes==0... depending on how
% the commit lick went.

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

top = sprintf('%s outcomes, %s strengths, %s stimulus: %i trials', os, strength2ana, stimrate2ana, sum(trs2ana));
disp(['Analyzing ', top])

%{
if strcmp(outcome2ana, 'all')    
    trs2ana = str2ana;
    os = sprintf('%s', outcome2ana);
else
    trs2ana = (outcomes==outcome2ana) & str2ana;
    os = sprintf('%i', outcome2ana);
end
%}

%%%
eventsToPlot = 'all'; % 10; % what events to align trials on and plot? % look at evT (below) to find the index of the event you want to plot.
if strcmp(eventsToPlot , 'all')
    ievents = 1:length(evT);
else
    ievents = eventsToPlot;
end


%%
% Set event times (ms) relative to when bcontrol starts sending the scope TTL. event times will be set to NaN for trs2rmv.
%{
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv);
%}

alldatanow = alldata(trs2ana);
trs2rmvnow = find(ismember(find(trs2ana), trs2rmv));
doplots = 0;


%% Align traces, average them, and plot them.

fn = sprintf('%s - Shades: stand error across trials (neuron-averaged traces of trials)', top);
% fn = sprintf('%s - Shades for rows 1 & 2: standard errors across neural traces (trial-averaged traces of neurons).  Shades for row 3: stand error across trials', top);
f = figure('name', fn);
cnt = 0;

for i = ievents
    
    cnt = cnt+1;
    
    % set DF/F traces (C: temporal component) and wheel traces
    
    disp(['------- ', evT{i}, ' -------'])
    
    eventTime = eval(evT{i});      
    % Take only trials in trs2ana for analysis
    if length(eventTime)>1
        eventTime = eventTime(trs2ana);
    end
    
    if ~iscell(eventTime) && isempty(eventTime), error('No trials!'), end    
    if (iscell(eventTime) && all(cellfun(@(x)all(isnan(x)), eventTime))) || (~iscell(eventTime) && all(isnan(eventTime)))
        warning('Choose outcome2ana and evT wisely! eg. you cannot choose outcome2ana=1 and evT="timeCommitIncorrResp"')
        error('Something wrong; eventTime is all NaNs. Could be due to improper choice of outcome2ana and evT{i}!')
    end
    
    traces = alldataDfofGood; % alldataSpikesGood; %  traces to be aligned.
    traces = traces(trs2ana);
    
    alignWheel = 1; printsize = 1;
    
    [traceEventAlign, timeEventAlign, nvalidtrs, traceEventAlign_wheelRev, ...
        timeEventAlign_wheelRev, nvalidtrs_wheel] = ...
        avetrialAlign_noTrGroup(eventTime, traces, alldatanow, frameLength, trs2rmvnow, alignWheel, printsize, doplots);
    
    %     disp('---------------------')
    %     figure, hold on
    %     plot(timeEventAlign_wheelRev, nvalidtrs_wheel(:,1))
    %     plot(timeEventAlign, nvalidtrs(:,1))
    
    
    %% plot wheel revolution
    
    subplot(3,length(ievents),length(ievents)*2+cnt), hold on
    
    tw = traceEventAlign_wheelRev;
%     tw = traceEventAlign_wheelRev(:,:,bb);
    top = nanmean(tw,3); % average across trials
    tosd = nanstd(tw,[],3);
    tosd = tosd / sqrt(size(tw,3)); % plot se
    
    if unitFrame
        e = find(timeEventAlign_wheelRev >= 0, 1);
        boundedline((1:length(top))-e, top, tosd, 'alpha')
    else
        boundedline(timeEventAlign_wheelRev, top, tosd, 'alpha')
    end
    %     plot(top)
    %     plot(timeEventAlign_wheelRev, top)
    
    xl1 = find(nvalidtrs_wheel >= round(max(nvalidtrs_wheel)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs_wheel >= round(max(nvalidtrs_wheel)*4/4), 1, 'last');
    
    if unitFrame
        xlim([xl1 xl2]-e)
    else
        xlim([timeEventAlign_wheelRev(xl1)  timeEventAlign_wheelRev(xl2)])
    end
    
    if unitFrame
        plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
        %         xlabel('Frame')
    else
        plot([frameLength/2  frameLength/2], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
        %         xlabel('Time')
    end
    ylim([min(top(xl1:xl2)-tosd(xl1:xl2))  max(top(xl1:xl2)+tosd(xl1:xl2))])
    %     ylabel('Wheel revolution')
    
    
    %% plot DF/F
    
    figure(f)
    subplot(3,length(ievents),length(ievents)*0+cnt), hold on
    
    % subplot(223), hold on
    av = squeeze(nanmean(traceEventAlign,2)); % frames x trials. (average across neurons).
%     av = nanmean(traceEventAlign,3); % frames x units. (average across trials).
    top = nanmean(av,2); % average across the other dimension (trials or neurons)
    tosd = nanstd(av, [], 2);
    tosd = tosd / sqrt(size(av, 2)); % plot se
    
    if unitFrame
        e = find(timeEventAlign >= 0, 1);
        boundedline((1:length(top))-e, top, tosd, 'alpha')
    else
        boundedline(timeEventAlign, top, tosd, 'alpha')
    end
    %     plot(top)
    %     plot(timeEventAlign, top)
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'last');
    
    if unitFrame
        xlim([xl1 xl2]-e)
    else
        xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
    end
    
    
    if unitFrame        
        plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
        %         xlabel('Frame')
    else
        plot([frameLength/2  frameLength/2], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
        %         xlabel('Time')
    end
    ylim([min(top(xl1:xl2)-tosd(xl1:xl2))  max(top(xl1:xl2)+tosd(xl1:xl2))])
    %     ylabel('DF/F')
    title(evT{i})
    
    
    
    %% set spikes (S)
    
    traces = alldataSpikesGood; % alldataDfofGood; %  traces to be aligned.
    % Take only trials in trs2ana for analysis
    traces = traces(trs2ana);
    
    alignWheel = 0; printsize = 0;
    
    [traceEventAlign, timeEventAlign, nvalidtrs] = ...
        avetrialAlign_noTrGroup(eventTime, traces, alldatanow, frameLength, trs2rmvnow, alignWheel, printsize, doplots);
    
    
    %% plot spikes
    %
    figure(f)
    subplot(3,length(ievents),length(ievents)*1+cnt), hold on
    
    % subplot(223), hold on
    av = squeeze(nanmean(traceEventAlign,2)); % frames x trials. (average across neurons).
%     av = nanmean(traceEventAlign,3); % frames x units. (average across trials).
    top = nanmean(av,2); % average across the other dimension (trials or neurons).
    tosd = nanstd(av, [], 2);
    tosd = tosd / sqrt(size(av, 2));  % plot se
    
    if unitFrame
        e = find(timeEventAlign >= 0, 1);
        boundedline((1:length(top))-e, top, tosd, 'alpha')
    else
        boundedline(timeEventAlign, top, tosd, 'alpha')
    end
    %     plot(top)
    %     plot(timeEventAlign, top)
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'last');
    
    if unitFrame
        xlim([xl1 xl2]-e)
    else
        xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
    end
    
    
    if unitFrame        
        plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
        %         xlabel('Frame')
    else
        plot([frameLength/2  frameLength/2],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
        %         xlabel('Time')
    end
    ylim([min(top(xl1:xl2)-tosd(xl1:xl2))  max(top(xl1:xl2)+tosd(xl1:xl2))])
    %     ylabel('Spiking')
    %     pause
    %}
end


subplot(3, length(ievents), length(ievents)*0+1)
ylabel('DF/F')

subplot(3, length(ievents), length(ievents)*1+1)
ylabel('Spiking')

subplot(3, length(ievents), length(ievents)*2+1)
ylabel('Wheel revolution')


if unitFrame
    subplot(3, length(ievents), 1)
    xlabel('Frame')
    subplot(3, length(ievents), length(ievents)*1+1)
    xlabel('Frame')
    subplot(3, length(ievents), length(ievents)*2+1)
    xlabel('Rotary sample (10ms)')
else
    subplot(3, length(ievents), length(ievents)*2+1)
    xlabel('Time')
end


%% Response side on early decision trials
%{
respEarly = NaN(1,length(f));

for ii=1:length(f),
    if ~isempty(alldata(f(ii)).parsedEvents.pokes.L) && isequal(alldata(f(ii)).parsedEvents.states.early_decision0(1), ...
            alldata(f(ii)).parsedEvents.pokes.L(end,1)),
%         'l',
        respEarly(ii) = 1;
    elseif ~isempty(alldata(f(ii)).parsedEvents.pokes.R) && isequal(alldata(f(ii)).parsedEvents.states.early_decision0(1), ...
            alldata(f(ii)).parsedEvents.pokes.R(end,1)),
%         'r',
        respEarly(ii) = 0;
    end,
    
    
end
%}

%% This is to  find the frame at which the sharp decrease happens for each trial.
%{
%% aligned on timeStop
av = squeeze(nanmean(traceEventAlign,2)); % frs x trs
av(1:349,:) = NaN;
d = diff(av);


%% find the frame at which the sharp decrease happens for each trial.
dd = d < -.05;
[fr,tr] = find(dd); 
figure; plot(tr,fr)
% figure; plot(nanmean(dd,2))


%%
% trials with the weird signal occuring late
aa = tr(tr>=73 & tr<162); 
aa(ismember(aa,trs2rmv))=[];

% trials with the weird signal occuring early
bb = tr(tr>162); 
bb(ismember(bb,trs2rmv))=[]; 

% 
% bb = tr(tr<73); 
% bb(ismember(bb,trs2rmv))=[];

%% 
% drop from 355 to 356
figure; plot(d(355,:))
% drop from 361 to 362
figure; plot(d(361,:))


%%
figure; 
x = true(1,162);
x(tr(tr<162)) = false;
% [nanmean(outcomes(x)==1), nanmean(allResp_HR_LR(x))]
% x = tr(tr<162);
subplot(211)
plot(outcomes(x))
hold on, plot(allResp_HR_LR(x))

x = true(1, size(d,2)-162);
x(tr(tr >= 162)) = false;
% [nanmean(outcomes(x)==1), nanmean(allResp_HR_LR(x))]
% x = tr(tr>=162);
subplot(212)
plot(outcomes(x))
hold on, plot(allResp_HR_LR(x))
%}


%% set lastSideLick
%{
find(outcomes==0, 1);

%%
% each element is time of stopRotary state (ms) for a trial.
stopT = arrayfun(@(x)x.parsedEvents.states.stop_rotary_scope(1), alldata)*1000;

% each cell shows time of sidelicks (ms) for a trial
allSideLicks = arrayfun(@(x)1000*vertcat(x.parsedEvents.pokes.R, x.parsedEvents.pokes.L), alldata, 'uniformoutput', 0);

% time of last side lick
lastSideLick = cellfun(@(x)max(x(:,1)), allSideLicks, 'uniformoutput',0);
lastSideLick(cellfun(@isempty, lastSideLick)) = {NaN};
lastSideLick = cell2mat(lastSideLick);

timeLastSideLickToStopT = stopT - lastSideLick;

%%

% trs2an = outcomes==0;
% timeLastSideLickToStopT = stopT(trs2an) - cell2mat(lastSideLick(trs2an));
%}


%%
%{
if strcmp(trials2ana , 'all')
    trs2ana = true(1, length(outcomes));
else
    s = abs(stimrate-cb)';
    allStrn = unique(s);
    
    if any(strcmp(trials2ana , {'incorr', 'incorrEasy', 'incorrHard', 'incorrMed'}))
        incorr = outcomes==0;
        incorrEasy = outcomes==0  &  (s >= (max(allStrn) - thStimStrength));
        incorrHard = outcomes==0  &  (s <= thStimStrength);
        incorrMed = outcomes==0  &  ((s > thStimStrength) & (s < (max(allStrn) - thStimStrength))); % intermediate strength
        %         fprintf('# trials: incorrEasy %i, incorrHard %i, incorrMed %i, (thStimStrength= %i)\n', sum(incorrEasy), sum(incorrHard), sum(incorrMed), thStimStrength)
        
    elseif any(strcmp(trials2ana , {'corr', 'corrEasy', 'corrHard', 'corrMed'}))
        corr = outcomes==1;
        corrEasy = outcomes==1  &  (s >= (max(allStrn) - thStimStrength));
        corrHard = outcomes==1  &  (s <= thStimStrength);
        corrMed = outcomes==1  &  ((s > thStimStrength) & (s < (max(allStrn) - thStimStrength))); % intermediate strength
        %         fprintf('# trials: corrEasy %i, corrHard %i, corrMed %i (thStimStrength= %i)\n', sum(corrEasy), sum(corrHard), sum(corrMed), thStimStrength)
    end
    
    trs2ana = eval(trials2ana);
end
fprintf('Analyzing %s, including %i trials.\n', trials2ana, sum(trs2ana))
%}


%{
load(imfilename, 'sdImage')
im = sdImage{2}; % ROIs will be shown on im
[CC2, mask2] = setCC_mask_manual(rois, im);


%%
a = nanmean(traceEventAlign,3); % average across trials
% f = find(((4089*3 + 956) - cs)<0, 1);
% a = squeeze(traceEventAlign(:,:,f)); % frames x units for trial f
xl1 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
xl2 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'last');
e = find(timeEventAlign>0, 1);

mn = min(min(a(xl1:xl2, :)));
mx = max(max(a(xl1:xl2, :)));



f1 = figure;
f2 = figure; imagesc(im), % axis image
for in = 1:size(a,2)
    % plot the trace
    figure(f1)
    set(gcf,'name',num2str(in))
    top = a(:,in);
    h = plot(top);
    hold on
    plot([e e],[min(top(xl1:xl2)) max(top(xl1:xl2))],'r:')
    xlim([xl1 xl2])
%     ylim([mn mx])
    
    % plot the roi
    figure(f2), hold on
    h = plot(CC2{in}(2,:), CC2{in}(1,:), 'color', 'r'); % colors(in, :));

    pause
    figure(f1), cla
    figure(f2), delete(h)
    
end

top = nanmean(a(:,ff),2);



% find neurons that are
% [70 320]
    956:959
    
% min(CC2{1}(1,:)) > 70
% max(CC2{1}(1,:)) < 320

mnn = cellfun(@(x)min(x(1,:)) > 70, CC2);
mxx = cellfun(@(x)max(x(1,:)) < 320, CC2);
ff = find(mnn==1 & mxx==1);
for in = ff
    figure(f2), hold on
    h = plot(CC2{in}(2,:), CC2{in}(1,:), 'color', 'r'); % colors(in, :));
end


ff = find(timeCommitCL_CR_Gotone > timeStimOffset+33);
a = squeeze(traceEventAlign(:,:,ff)); % frames x units for trial f
top = nanmean(nanmean(a,2),3);

a = squeeze(traceEventAlign); %
top = nanmean(nanmean(a,2),3);


%% make ROIs outside the neurons in a dark area

clear rois
i = 1;
[X,Y] = meshgrid(66:90, 1:45);
rois{i}.mnCoordinates = [Y(:), X(:)]';

i = 2;
[X,Y] = meshgrid(373:398, 1:30);
rois{i}.mnCoordinates = [Y(:), X(:)]';

i = 3;
[X,Y] = meshgrid(195:210, 270:295);
rois{i}.mnCoordinates = [Y(:), X(:)]';

i = 4;
[X,Y] = meshgrid(244:260, 409:440);
rois{i}.mnCoordinates = [Y(:), X(:)]';


for in = 1:4
    hold on
    h = plot(rois_custom2{in}.mnCoordinates(2,:), rois_custom2{in}.mnCoordinates(1,:), 'color', 'r'); % colors(in, :));
end


activity = activity_custom2;
dFOF = konnerthDeltaFOverF(activity, pmtOffFrames{gcampCh}, smoothPts, minPts);
    
%}
