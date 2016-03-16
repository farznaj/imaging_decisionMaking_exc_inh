function [traceEventAlign, timeEventAlign, nvalidtrs, traceEventAlign_wheelRev,...
    timeEventAlign_wheelRev, nvalidtrs_wheel] = avetrialAlign_noTrGroup(eventTime, traces, alldata, frameLength, trs2rmv, doplots)

% eventTime = time1stCenterLick; timeInitTone; % 'beg'; timeStimOnset; % timeStimOnset; % timeReward;  % eventTime = cellfun(@(x)x(1),timeInitTone); %timeInitTone; % first trial initi tone.
% traces =  alldataSpikesGood; % alldataDfofGood; % traces to be aligned.

if ~exist('doplots', 'var')
    doplots = false;
end


%% Align traces on eventTimes. (the earlier method which doesn't require nPre and nPost frames and will include all avaialble frames before and after the event of interest. But later you need to zoom just on frames with a large number of trials contributing to them.)
traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.
shiftTime = frameLength / 2;
scaleTime = frameLength;

% find the frame during which the event of interest (eg. time1stCenterLick) happened for each trial.
if length(eventTime)==1 && eventTime==1
    % align on the first frame (when scanning started, this may make more sense than trial init. tone bc mice seem to use this bc of shutter sound).
    eventInds_f = ones(size(alldata));
    eventInds_f(trs2rmv) = NaN;
else
    eventInds_f = eventTimeToIdx(eventTime, traceTimeVec); % index of eventTime on traceTimeVec array for each trial.
end


% Align dfof traces on particular events (like time1stCenterLick).
% traceEventAlign: frames x units x trials
% timeEventAlign: 1 x frames
[traceEventAlign, timeEventAlign] = triggerAlignTraces(traces, eventInds_f, shiftTime, scaleTime); % frames x units x trials.



% some checks
%{
numTrsWevent = sum(~isnan(eventInds_f))
mnFrame0_mxFrame0_mxNumFrames = [min(eventInds_f) max(eventInds_f) max(framesPerTrial)]
eventFrameNum = max(eventInds_f) % find(timeEventAlign==frameLength/2) % in what frame of traceEventAlign, the event exists.
%}
if ~isequal(max(eventInds_f), find(timeEventAlign==frameLength/2)), error('something wrong!'); end

nvalidtrs = sum(~isnan(traceEventAlign),3); % frames x neurons; number of trials that contribute to each frame for each neuron.
nvalidtrs = nvalidtrs(:,1);
% figure; subplot(221), plot(nvalidtrs(:,1)) % shows at each frame of traceEventAlign how many trials are contributing to the average.
% xlabel('Frame')
% ylabel('# Contributing trials')

fprintf('%d %d %d = size(traceEventAlign)\n', size(traceEventAlign)) % frames x units x trials.



% traceTimeVec = framet;
% eventInds = frame0s;
%{
if iscell(eventTime)
    eventTime = cellfun(@(x)x(1), eventTime);
end
% frametimes and eventTime are relatie to bcontrol ttl onset. so we compare them to find on what frame eventTime happened.
framet = {alldata.frameTimes};
frame0s = findFrame0(framet, eventTime);
%}

%{
%% Align dfof traces on particular events (like time1stCenterLick).
shiftTime = frameLength / 2;
scaleTime = frameLength;
% frames x units x trials
% in timeEventAlign frame0 (ie the frame to align trials on) will be 0+frameLength/2.
% in traceEventAlign, traces are shifted by max(frame0s) - frame0s(itr)
[traceEventAlign, timeEventAlign] = triggerAlignTraces(alldataDfofGood, frame0s, shiftTime, scaleTime);
%}
% spikes
% [traceEventAlign_s, timeEventAlign_s] = triggerAlignTraces(alldataSpikesGood, frame0s, shiftTime, scaleTime);


%% Align wheelRev on eventTime

wheelTimeRes = alldata(1).wheelSampleInt;
shiftTime = wheelTimeRes / 2;
scaleTime = wheelTimeRes;
[traces_wheel, times_wheel] = wheelInfo(alldata);
% shiftTime = times_wheel{1}(1);
% scaleTime = times_wheel{1}(1)*2;

% find the frame during which the event of interest (eg. time1stCenterLick) happened.
if length(eventTime)==1 && eventTime==1
    eventInds_w = ones(size(alldata));
    eventInds_w(trs2rmv) = NaN;
else
    eventInds_w = eventTimeToIdx(eventTime, times_wheel); % index of eventTime on traceTimeVec array for each trial.
end

% Align dfof traces on particular events (like time1stCenterLick).
[traceEventAlign_wheelRev, timeEventAlign_wheelRev] = triggerAlignTraces(traces_wheel, eventInds_w, shiftTime, scaleTime);

% some checks
%{
numTrsWevent = sum(~isnan(eventInds_w))
mnFrame0_mxFrame0_mxNumFrames = [min(eventInds_w) max(eventInds_w) max(framesPerTrial)]
%}
fprintf('%d %d %d = size(traceEventAlign_wheelRev)\n', size(traceEventAlign_wheelRev)) % timeSamples x 1 x trials.

% compute how many trials contribute to each time point of
% timeEventAlign_wheelRev
nvalidtrs_wheel = sum(~isnan(traceEventAlign_wheelRev),3); % timeSamples x 1; number of trials that contribute to each frame for each neuron.
nvalidtrs_wheel = nvalidtrs_wheel(:,1);
% subplot(222), plot(nvalidtrs_wheel(:,1)) % shows at each frame of traceEventAlign how many trials are contributing to the average.

% the following two should have small difference (eventTime, in ms, based on
% wheelRev trace and ca trace): 
% remember the max rotary array is defined in arduino 25000points (ms), so
% if you see a big difference here it should be bc wheelRev reached its max
% length.
fprintf('%.3f = diff of eventTime btwn imaging and wheelRev\n', diff([max(eventInds_f)*frameLength  max(eventInds_w)*wheelTimeRes]))

% find where in wheelTimes, eventTime occurred.
% bint = {alldata.wheelTimes};
% bin0s = findFrame0(bint, eventTime);


% subplot(223), plot(timeEventAlign_wheelRev, nvalidtrs_wheel(:,1))
% hold on; plot(timeEventAlign, nvalidtrs(:,1))





%% start plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if doplots
    %%
    % perhaps exclude some trs at the beg w extra stim, etc. you need to know
    % what's going in the average.
    
    % exclude parts of the trace w too few trials.
    
    % you need to look at the trace at a good zoom level
    
    % you perhaps need to find the baseline preced stim_on, only include trs w
    % stable baseline (preced stim_on), and shift the baselines to the same
    % value so you can make the comparison between lo and hi rate traces.
    
    
    
    %% Prepare vars for plotting
    
    aveTrsPerNeuron = nanmean(traceEventAlign,3); % frames x units, averaged across all trials. aligned on event time.
    
    eventFrameNum = max(eventInds_f);
    respWinMs = [0 500]; % 0 to 500ms after the event we will look for a response.
    respWin = eventFrameNum + (round(respWinMs(1)/frameLength):round(respWinMs(2)/frameLength));
    
    
    %%
    sdTrsPerNeuron = nanstd(traceEventAlign,0,3);
    seTrsPerNeuron = sdTrsPerNeuron ./ repmat(sqrt(nvalidtrs), [1 size(sdTrsPerNeuron,2)]);
    % aveTrsPerNeuron_S = nanmean(traceEventAlign_s,3);
    
    frs2plot = nvalidtrs >= max(nvalidtrs)-5; 2;
    t = timeEventAlign(frs2plot);
    plotwheelRev = true; false;
    plotmarkers = false;
    
    
    if plotwheelRev
        aveTrsPerNeuron_wheel = nanmean(traceEventAlign_wheelRev,3); % average wheel movement across all trials aligned on eventTime.
        frs2plot_wheel = nvalidtrs_wheel >= max(nvalidtrs_wheel);
    end
    
    
    blDfof = quantile(dfofGood, .1);
    if plotmarkers
        % Compute mean and std of peaks of the dfofGood trace for each neuron.
        peaksDfofTrace = cell(1, sum(goodNeurons));
        for ineu = 1:sum(goodNeurons)
            [pks,~, ~, ~] = findpeaks(dfofGood(:,ineu), 'minpeakheight', blDfof(ineu)+.05); %'minpeakdistance', 3, 'threshold', .005);
            peaksDfofTrace{ineu} = pks;
        end
        quant80Peaks = cellfun(@(x)quantile(x, .8), peaksDfofTrace);
    end
    
    
    %% set gaussian filter
    dofilter = 0; % 1;
    
    if dofilter
        siz = 5;
        sig = 1;
        x = -floor(siz/2) : floor(siz/2);
        H = exp(-(x.^2/ (2 * sig^2)));
        H = H/sum(H);
    end
    
    
    %% plot traces per neuron (averaged across trials).
    
    figure;
    cnt = 0;
    
    for ineu = 1:size(traceEventAlign,2) % i_ttestp; % randperm(size(traceEventAlign,2)) %
        cnt = cnt+1;
        hold on
        
        toplot = aveTrsPerNeuron(frs2plot, ineu);
        %     toplot_b = sdTrsPerNeuron(frs2plot, ineu);
        toplot_b = seTrsPerNeuron(frs2plot, ineu);
        
        if dofilter
            toplot = conv(toplot, H');
            toplot_b = conv(toplot_b, H);
            
            toplot = toplot(ceil(siz/2) : end-ceil(siz/2)+1);
            toplot_b = toplot_b(ceil(siz/2) : end-ceil(siz/2)+1);
        end
        
        %     plot(t, toplot, 'color', [114 189 255]/256)
        %     plot(t, toplot+toplot_sd, ':', 'color', [114 189 255]/256)
        %     plot(t, toplot-toplot_sd, ':', 'color', [114 189 255]/256)
        
        boundedline(t, toplot, toplot_b)
        
        %     plot(t, aveTrsPerNeuron_S(frs2plot, ineu))
        
        %% plot the average wheel movement across trials aligned on eventTime.
        if plotwheelRev
            plot(timeEventAlign_wheelRev(frs2plot_wheel), -aveTrsPerNeuron_wheel(frs2plot_wheel), 'k:')
        end
        
        
        %% put some markers
        plot([t(1) t(end)], [blDfof(ineu)  blDfof(ineu)], 'r:')
        if plotmarkers
            % bl_th
            bl_th = quant80Peaks(ineu);
            plot([t(1) t(end)], [bl_th  bl_th], 'r:')
            
            
            % a line at 0 (ie the event time)
            plot([0 0], [min(toplot) max(toplot)])
        end
        
        %% response window
        plot([respWinMs(1)  respWinMs(1)], [min(toplot-toplot_b) max(toplot+toplot_b)], 'g-.')
        plot([respWinMs(2)  respWinMs(2)], [min(toplot-toplot_b) max(toplot+toplot_b)], 'g-.')
        
        
        %%
        %     mn = min(bl_th, blDfof(ineu));
        %     mx = max([bl_th; blDfof(ineu); toplot(:)]);
        %     ylim([mn mx])
        
        xlim([-500 500])
        xlabel('Time since event onset (ms)')
        ylabel('DF/F')
        set(gca, 'tickdir', 'out')
        set(gcf, 'name', sprintf('Neuron %d', ineu))
        %     set(gcf, 'name', sprintf('Neuron %d, P=%.2f', ineu, s_ttestp(cnt)))
        
        %%
        pause
        delete(gca)
        
    end
    
    
    
    
    
    
    
    
    %% group based on stim rate
    usr = unique(stimrate);
    trs2ana_lo = find(ismember(stimrate, [usr(1:3)]));
    trs2ana_hi = find(ismember(stimrate, [usr(end-2:end)]));
    
    % find(~[alldata.extraStimDuration], 1) % trs w extra stim that need to be excluded
    % also find what else you do at the begining and exclude those trials too.
    
    aveTrsLoPerNeur = NaN(size(traceEventAlign,1), size(traceEventAlign,2));
    aveTrsHiPerNeur = NaN(size(traceEventAlign,1), size(traceEventAlign,2));
    for ineu = 1:size(traceEventAlign,2)
        aveTrsLoPerNeur(:,ineu) = nanmean(traceEventAlign(:, ineu, trs2ana_lo), 3);
        aveTrsHiPerNeur(:,ineu) = nanmean(traceEventAlign(:, ineu, trs2ana_hi), 3);
    end
    
    
    %%
    figure;
    for ineu = 1:size(traceEventAlign,2)
        hold on
        plot(timeEventAlign, aveTrsLoPerNeur(:, ineu))
        plot(timeEventAlign, aveTrsHiPerNeur(:, ineu))
        
        legend('loRate','hiRate')
        pause
        delete(gca)
    end
    
    
end


%%
%{
toplot = group1_aveTrace;

figure; hold on
imagesc(toplot)
plot([eventFrameNum eventFrameNum],[1 size(toplot,1)])
ylim([1 size(toplot,1)])
xlim([1 size(toplot,2)])

%%
figure; hold on
p1 = plot(timeEventAlign, nanmean(group1_ave,2));
p2 = plot(timeEventAlign, nanmean(f_group2,2));
plot([0 0],[-1 2], 'color', [.6 .6 .6])
% ylim([0 1.2])
xlim([-15000 15000])
pause
delete([p1,p2])

for n = 1:size(group1_aveTrace)
    f = find(timeEventAlign>0, 1);
    baselineIndx = max(1,f-round(100/frameLength)):max(1,f-1);
    basef_g1 = nanmean(f_group1(baselineIndx,n));
    basef_g2 = nanmean(group2_ave(baselineIndx,n));
    basef_g1 = 0;
    basef_g2 = 0;
    
    hl = plot(timeEventAlign, f_group1(:,n)-basef_g1, 'color', [.6 .6 .6]);
    hh = plot(timeEventAlign, group2_ave(:,n)-basef_g2, 'k');
    title(n)
    
    ff = round(10000/frameLength);
    yl2 = max([f_group1(max(1,f-ff):f+ff,n)-basef_g1; ...
        group2_ave(max(1,f-ff):f+ff,n)-basef_g2]);
    
    yl1 = min([f_group1(max(1,f-ff):f+ff,n)-basef_g1;...
        group2_ave(max(1,f-ff):f+ff,n)-basef_g2]);
    
    ylim([yl1-.03 yl2+.03])
    
    pause
    delete([hl hh])
end
%}


