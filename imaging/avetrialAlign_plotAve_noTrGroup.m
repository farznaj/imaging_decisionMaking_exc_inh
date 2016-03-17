% This script plots average traces across all neurons and all trials
% aligned on particular trial events. 
% It calls avetrialAlign_noTrGroup which calls triggerAlignTraces for
% alignment : you don't specify prePost frames, it uses NaNs and takes all
% frames into the average. Then you only look at portion of the trace that
% has enough number of trials contributing to the average.


%%
evT = {'1', 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',... 
'timeReward', 'timeCommitIncorrResp'}; %,... 
% 'centerLicks', 'leftLicks', 'rightLicks'};

% time1stCenterLick, time1stCorrectResponse, timeStop
% eventTime = cellfun(@(x)x(1),timeInitTone); %timeInitTone; % first trial initi tone.


%%
f = figure;
doplots = 0;

for i = 1:length(evT)

    eventTime = eval(evT{i});       
    traces = alldataDfofGood; % alldataSpikesGood; %  traces to be aligned.
    
    [traceEventAlign, timeEventAlign, nvalidtrs, traceEventAlign_wheelRev, ...
        timeEventAlign_wheelRev, nvalidtrs_wheel] = ...
        avetrialAlign_noTrGroup(eventTime, traces, alldata, frameLength, trs2rmv, doplots);
    
%     figure, hold on
%     plot(timeEventAlign_wheelRev, nvalidtrs_wheel(:,1))
%     plot(timeEventAlign, nvalidtrs(:,1))
    
    
    %% plot wheel revolution

    subplot(3,length(evT),length(evT)*2+i), hold on
    top = nanmean(traceEventAlign_wheelRev,3);
    plot(top)
%     plot(timeEventAlign_wheelRev, top)
    
    xl1 = find(nvalidtrs_wheel >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs_wheel >= round(max(nvalidtrs)*3/4), 1, 'last');
    
    xlim([xl1 xl2])
%     xlim([timeEventAlign_wheelRev(xl1)  timeEventAlign_wheelRev(xl2)])
    xlabel('time')
    ylabel('wheel revolution')
    
%     plot([0 0],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    e = find(timeEventAlign_wheelRev > 0, 1);
    plot([e e], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    
    
    %% plot DF/F

    figure(f)
    subplot(3,length(evT),length(evT)*0+i), hold on
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
    xlabel('time')
    ylabel('DF/F')
    title(evT{i})
    
    
    %%
    traces = alldataSpikesGood; % alldataDfofGood; %  traces to be aligned.
    [traceEventAlign, timeEventAlign, nvalidtrs, traceEventAlign_wheelRev, ...
        timeEventAlign_wheelRev, nvalidtrs_wheel] = ...
        avetrialAlign_noTrGroup(eventTime, traces, alldata, frameLength, trs2rmv, doplots);
    
    
    %% plot spikes

    figure(f)
    subplot(3,length(evT),length(evT)*1+i), hold on
    % subplot(223), hold on
    top = nanmean(nanmean(traceEventAlign,3),2); % average across trials and neurons.
    plot(top)
%     plot(timeEventAlign, top)
    % plot(timeEventAlign_wheelRev, -nanmean(traceEventAlign_wheelRev,3), 'k:')
    
    xl1 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
    xl2 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'last');
    
%     plot([0 0],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    e = find(timeEventAlign > 0, 1);
    plot([e e], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
    
    xlim([xl1 xl2])
%     xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
    xlabel('time')
    ylabel('Spiking')
    
%     pause
    
end



%%
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
