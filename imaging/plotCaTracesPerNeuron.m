function plotCaTracesPerNeuron(traceFU_toplot, alldataDfofGood, alldataSpikesGood, interactZoom, plotTrs1by1, markQuantPeaks, showitiFrNums, varargin)
% plotCaTracesPerNeuron(traceFU_toplot, interactZoom, plotTrs1by1, markQuantPeaks, alldataDfofGood, alldataSpikesGood, showitiFrNums, varargin)
% varargin: {framesPerTrial, alldata, S_mcmc, dFOF_man, eftMatchIdx_mask, allEventTimes, stimrate}) 
%
% allEventTimes = {timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, timeStop, centerLicks, leftLicks, rightLicks};
%
% traces are required in format frames x units
%
% interactZoom = 1; % if 1, zoom will be active when watching neurons 1-by-1.
% plotTrs1by1 = 1; % plot trials of each neuron one by one.
% markQuantPeaks = 0; % if 1, peak quantiles will be plotted on the trace.
%
% run aveTrialAlign to get variables for this script.

% traceFU_toplot = alldataDfofGood_mat; % FU means frame x unit

%{
addNaNiti = 1; % if 1, a NaN will be added to the trace in between trials. % if 1 you need to provide framesPerTrial
showitiFrNums = 1; % if 1, number of frames that are missing between trials will be shown on the plot. % if 1 you need to provide alldata
compareManual = 1; % traces from manual method will be also superimposed. You need to provide dFOF_man and eftMatchIdx_mask_good and framesPerTrial
%}

% traceFU_toplot = cell2mat(alldataDfofGood'); % frames x units
% allTimes = [alldata.frameTimes]';
if ~isempty(varargin)
    if ~isempty(varargin{1}{1})
        framesPerTrial = varargin{1}{1};
        addNaNiti = 1;
    else
        addNaNiti = 0;
    end
    
    if length(varargin{1})>1 && ~isempty(varargin{1}{2})
        alldata = varargin{1}{2};
%         showitiFrNums = 1;
    else
%         showitiFrNums = 0;
    end
    
    if length(varargin{1})>2 && ~isempty(varargin{1}{3})
        strace = varargin{1}{3};
        showSpikes = 1;
    else
        showSpikes = 0;
    end
    
    if length(varargin{1})>3 && ~isempty(varargin{1}{4}) && ~isempty(varargin{1}{5})
        dFOF_man = varargin{1}{4};
        eftMatchIdx_mask = varargin{1}{5};
        compareManual = 1;
    else
        compareManual = 0;
    end
    
    if length(varargin{1})>5 && ~isempty(varargin{1}{6}) && ~isempty(varargin{1}{7})
        timeInitTone = varargin{1}{6}{1};
        timeStimOnset = varargin{1}{6}{2};
        timeStimOffset = varargin{1}{6}{3};
        timeCommitCL_CR_Gotone = varargin{1}{6}{4};
        time1stSideTry = varargin{1}{6}{5};
        timeReward = varargin{1}{6}{6};
        timeCommitIncorrResp = varargin{1}{6}{7};
        timeStop = varargin{1}{6}{8};
        centerLicks = varargin{1}{6}{9};
        leftLicks = varargin{1}{6}{10};
        rightLicks = varargin{1}{6}{11};
        
        stimrate = varargin{1}{7};
        
        markEventTimes = 1;
    else
        markEventTimes = 0;
    end
    %{
    if length(varargin{1})>7 && ~isempty(varargin{1}{8})
        NsToPlot = varargin{1}{8}; % badQual; % 1:size(traceFU_toplot{1},2)
    else
        NsToPlot = 1:size(traceFU_toplot{1},2);
    end
    %}
else
    addNaNiti = 0;
    showitiFrNums = 0;
    compareManual = 0;
    showSpikes = 0;
    markEventTimes = 0;
    NsToPlot = 1:size(traceFU_toplot{1},2);
end


%%
if exist('framesPerTrial', 'var')
    framesPerTr = framesPerTrial(~isnan(framesPerTrial));
    % framesPerTr = cellfun(@(x)size(x,1), alldataDfofGood); % compared to framesPerTrial, framesPerTr doesn't have NaN at the end bc the last trial is excluded.
% end

% if compareManual || plotTrs1by1
    csfrs = [0 cumsum(framesPerTr)];
    begf = csfrs+1; % first frame of each trial.
    
    if addNaNiti
        % find the new indeces of csfrs in the traces that have NaNs inserted.
%         csfrs = csfrs + (0:length(csfrs)-1);
        begf = (csfrs + (0:length(csfrs)-1)) + 1; % % find the new indeces of csfrs in the traces that have NaNs inserted.
    end
    
    %     lastTr2plot = find((csfrs-size(traceFU_toplot{1},1))>0, 1)-2; % use this if you are using part of a trace not the entire trace.
    lastTr2plot = length(alldataDfofGood);
end


%% add NaN between trials.
if addNaNiti
    begFrames = csfrs+1; % first frame of each trial.
    
    for itp = 1:length(traceFU_toplot)
        begFrames(begFrames>length(traceFU_toplot{itp})) = []; % this is in case you are plotting part of the entire movie.
        % remember in insertElement, inds 2 be added are relative to the
        % original array.
        traceFU_toplot{itp} = insertElement(traceFU_toplot{itp}, begFrames(2:end), NaN); % insert NaN right before the beginning of each trial.
        %         traceFU_toplot{itp}(begFrames(2:end)-1,:) = NaN;
    end
    % activityGood_nan(begFrames(2:end)-1,:) = NaN;
    
    if compareManual
        dFOF_man = insertElement(dFOF_man, begFrames(2:end), NaN);
    end
    
    if showSpikes
        strace = insertElement(strace, begFrames(2:end), NaN);
    end
end


%%
if plotTrs1by1
    labels = [1 0 -1 -2 -3 -4 -5];
    labels_name = {'success', 'failure', 'early decision', 'no decision', 'wrong start', 'no center commit', 'no side commit'};
end

if showSpikes
%     sth = .3; % threshold to identify spikes on strace.
    a = boxFilterNaN(nanmean(strace,2), 200);
    sth = quantile(a, .1); % use the lowest 10 percentile on the smoothed average trace as the threshold for identifying spikes.
    
    snan = nan(size(strace));
    snan(strace > sth) = 1;
    
    if exist('alldataSpikesGood', 'var')
        snantr = cell(1, length(alldataDfofGood));
        for itr = 1:length(alldataDfofGood)
            snantr{itr} = nan(size(alldataSpikesGood{itr}));
            snantr{itr}(alldataSpikesGood{itr} > sth) = 1;
        end
    end
end


%% Preceding iti (sec)

if showitiFrNums
    
    [iti_noscan, iti_gui_state] = itiSet(alldata); % ms
    frameLength = 1000/30.9; % msec.
    nFrames_iti = round(iti_noscan/frameLength); % nFrames between trials (during iti) that was not imaged.

    lab_iti = num2str(nFrames_iti);

    figure; plot(iti_gui_state/1000)
    hold on, plot(iti_noscan/1000)    
    xlabel('Trial number')
    ylabel('ITI (s)')
    legend('set in GUI','computed from states','no scanning')
            
end


%% Compute mean and std of peaks of the dfofGood trace for each neuron.
if markQuantPeaks || plotTrs1by1
    
    peaksDfofTrace = cell(1, size(traceFU_toplot{1},2)); % remember if it is C_df it has the baseline in it too so you need to take it out otherwise number of units will be 1 more.
    %     blDfof = quantile(dfofGood, .1);
    blDfof = quantile(traceFU_toplot{1}, .1);
    for ineu = 1:size(traceFU_toplot{1},2)
        %         [pks,~, ~, ~] = findpeaks(dfofGood(:,ineu), 'minpeakheight', blDfof(ineu)+.05); %'minpeakdistance', 3, 'threshold', .005);
        [pks,~, ~, ~] = findpeaks(traceFU_toplot{1}(:,ineu), 'minpeakheight', blDfof(ineu)+.05); %'minpeakdistance', 3, 'threshold', .005);
        
        peaksDfofTrace{ineu} = pks;
    end
    
    avePeaks = cellfun(@mean, peaksDfofTrace);
    sdPeaks = cellfun(@std, peaksDfofTrace);
    quant75Peaks = cellfun(@(x)quantile(x, .75), peaksDfofTrace);
    quant80Peaks = cellfun(@(x)quantile(x, .8), peaksDfofTrace);
    quant85Peaks = cellfun(@(x)quantile(x, .85), peaksDfofTrace);
end


%% Set the baseline preceding a specific event for each trial and each neuron
% if traceEventAlign is aligned on timeInitTone, so bl would be frame at
% which timeInitTone is -3:-1.
if plotTrs1by1
    
    eventTime = timeInitTone;
    if iscell(eventTime)
        eventTime = cellfun(@(x)x(1), eventTime);
    end
    
    %     blPrecedEvent = NaN(size(alldataDfofGood{1},2), length(alldataDfofGood)); % neurons x trials
    blPrecedEvent = NaN(size(traceFU_toplot{1},2), length(alldataDfofGood)); % length(framesPerTr));
    % figure;
    for itr = 1: length(alldataDfofGood) % find((csfrs-size(traceFU_toplot{1},1))>0, 1)-2; % size(blPrecedEvent,2) % the 1st is prefered in case you don't have trace of the entire session
        if ~isnan(eventTime(itr))
            %     f = ceil(timeInitTone{itr}(1)/frameLength)-1;
            f = findFrame0({alldata(itr).frameTimes}, eventTime(itr)); % frame at which timeInitTone occured. Remember this frame index corresponds to dfof and not traceEventAlign.
            blRange = f-floor(100/frameLength) : f-1; % compute baseline over 100ms window preceding timeInitTone (only a few frames)
            %     blRange = 1:f; % all frames preceding toneInit
            
            for ineu = 1:size(blPrecedEvent,1)
                trace = alldataDfofGood{itr}(:, ineu); % traceFU_toplot{1}(csfrs(itr)+1:csfrs(itr+1), ineu);
                blPrecedEvent(ineu,itr) = nanmean(trace(blRange));
                %                 blPrecedEvent(ineu,itr) = nanmean(alldataDfofGood{itr}(blRange,ineu));
                
                %{
            hold on
            top = alldataDfofGood{itr}(:,ineu);
            plot(alldataDfofGood{itr}(:,ineu))
            plot([blRange(1) blRange(1)], [min(alldataDfofGood{itr}(:,ineu)) max(alldataDfofGood{itr}(:,ineu))])
            plot([blRange(end) blRange(end)], [min(alldataDfofGood{itr}(:,ineu)) max(alldataDfofGood{itr}(:,ineu))])
            plot([0 length(top)], [blPrecedEvent(ineu,itr) blPrecedEvent(ineu,itr)])
%             plot([0 length(top)], [avePeaks(ineu) avePeaks(ineu)])
            plot([0 length(top)], [quant75Peaks(ineu) quant75Peaks(ineu)])
            
            pause
            delete(gca)
                %}
            end
        end
    end
    
    figure; imagesc(blPrecedEvent)
    colorbar
    xlabel('Trial number')
    ylabel('Neuron')
    title('Baseline preceding initiation tone')
end


%% Turn alldataDfofGood to a matrix of size frames x units x trials
%{
alldataDfofGood_fut = NaN(max(framesPerTr), size(alldataDfofGood{1},2), length(alldataDfofGood)); % frames x units x trials     % _fut indicates fr x units x trs
for itr = 1:length(alldataDfofGood)
    for ineu = 1:size(alldataDfofGood_fut,2)
        alldataDfofGood_fut(1:size(alldataDfofGood{itr},1), ineu, itr) = alldataDfofGood{itr}(:,ineu);
    end
end
size(alldataDfofGood_fut)

% figure; plot(alldataDfofGood_fut(:,43,23))
% hold on, plot(alldataDfofGood{23}(:,43))
%}


%%
f1 = figure;
set(axes,'position',[0.0338    0.1815    0.9389    0.6546])
set(gcf,'color','w')
set(gca,'xaxislocation','top', 'ticklength', [.002 0])
if plotTrs1by1
    f2 = figure;
end


for ineu = 1:size(traceFU_toplot{1},2) % NsToPlot % badQual;
    
    figure(f1)
    set(f1,'name', sprintf('Neuron %d', ineu))
    %     set(gca, 'tickdir','out', 'NextPlot', 'replacechildren');
    hold on
    h = []; mn = []; mx = [];
    
    % raw activity
    %     toplot = activityGood_nan(:,ineu);
    %     toplot = toplot - quantile(toplot, .2);
    %     toplot = toplot /range(toplot);
    %     h1 = plot(toplot,'g');
    
    
    if compareManual
        imatched = eftMatchIdx_mask(ineu);
        
        hn = plot(dFOF_man(:, imatched));
        h = [h hn];
        mn = [mn min(dFOF_man(:, imatched))];
        mx = [mx max(dFOF_man(:, imatched))];
    end
    
    for itp = 1:length(traceFU_toplot)
        toplot = traceFU_toplot{itp}(:,ineu);
        %     toplot = toplot - quantile(toplot, .2);
        %     toplot = toplot /range(toplot);
        hn = plot(toplot);
        h = [h hn];
        mn = [mn min(toplot)];
        mx = [mx max(toplot)];
    end
    
    if ~all(isnan(toplot))
        
    mn = min(mn);
    mx = max(mx);
    
    if showSpikes
        hn = plot(snan(:, ineu)*mx, 'k.', 'markersize', 10);
        h = [h hn];
    end
    
    %     mn = min(cellfun(@(x)min(x(:,ineu)), traceFU_toplot)); % min(toplot);
    %     mx = max(cellfun(@(x)max(x(:,ineu)), traceFU_toplot)); % max(toplot);
    % plot lines indicating the begining of each trial.
    %     plot([begFrames; begFrames], repmat([mn;mx], 1, length(begFrames)), 'color', [.6 .6 .6])
    %     if compareManual
    %         mn = min(mn, min(dFOF_man(:, imatched)));
    %         mx = max(mx, max(dFOF_man(:, imatched)));
    %     end
    box off
    xlim([1 length(toplot)])
    ylim([mn-.1 mx+.1])
    set(gca,'tickdir','out')
    
    
    %% for each iti show how many frames the non-recorded iti equals to.
    if showitiFrNums        
        x = begf(2:end)-1;
%         x = begFrames(2:end)-1;
        t = NaN(length(alldataDfofGood), 2); % NaN(length(lab_iti)-1,2); % find((csfrs-size(traceFU_toplot{1},1))>0, 1)-2
        for iiti = 1:length(begf)-2 % length(alldataDfofGood)-1 % find((csfrs-size(traceFU_toplot{1},1))>0, 1)-2; % length(lab_iti)-1
            t(iiti,1) = text(x(iiti), mn-.2, num2str(iiti+1)); % trial number of the following trace.
            t(iiti,2) = text(x(iiti), mn-.45, lab_iti(iiti+1,:)); % non-imaged frames preceding the following trace.
        end
    end
    
    
    %% Mark avePeaks to find unstable baselines.
    %     plot([0 length(toplot)], [avePeaks(ineu) avePeaks(ineu)])
    %     plot([0 length(toplot)], [avePeaks(ineu) + sdPeaks(ineu)*1.2   avePeaks(ineu) + sdPeaks(ineu)*1.2])
    if markQuantPeaks
        hn = plot([0 length(toplot)], [quant75Peaks(ineu)   quant75Peaks(ineu)], 'g');
        h = [h hn];
        
        hn = plot([0 length(toplot)], [quant80Peaks(ineu)   quant80Peaks(ineu)], 'b');
        h = [h hn];
        
        hn = plot([0 length(toplot)], [quant85Peaks(ineu)   quant85Peaks(ineu)], 'c');
        h = [h hn];
    end
    
    
    %% look closely at the trace.
    if interactZoom
        %         xlim([1 1e4])
        %         xl = [1 1e4];
        %         numSect = ceil(length(toplot) / xl(2));
        
        %         for isect = 1:numSect
        %
        %             xlim([(isect-1)*xl(2)+1-100  isect*xl(2)])
        %             zoom on
        %             zoom reset
        z = zoom;
        set(z,'Motion','horizontal','Enable','on');
        
        %%
        %             if isect~=numSect
        %                 pause
        %             end
        %
        %         end
        %         zoom off
    end
    
    %% plot average of trials per neuron
    %{
    figure(f2)
    toplot = nanmean(squeeze(alldataDfofGood_fut(:,ineu, :)),2);
    plot(toplot)
    %}
    
    
    %% plot trials of each neuron one by one.
    if plotTrs1by1
        
        %     bl_stable = quantile(traceFU_toplot(:,ineu), .4);
        bl_th = quant80Peaks(ineu); % 1.5; % bl_stable*8 % *5 needs assessment. % you need to know what the amp of ca transients is, and then determin based on that what bl_th should be, eg you may be fine w 30% of that..
        
        loBlTrs = find(blPrecedEvent(ineu,:) <= bl_th);
        hiBlTrs = find(blPrecedEvent(ineu,:) > bl_th);
        len_loHi = [length(loBlTrs)  length(hiBlTrs)];
        
        for itr = 1: length(alldataDfofGood) % find((csfrs-size(traceFU_toplot{1},1))>0, 1)-2 % find((csfrs-size(traceFU_toplot{1},1))>0, 1)-2; % length(framesPerTr) % length(alldataDfofGood) % loBlTrs % hiBlTrs; % 1:size(alldataDfofGood_fut,3)
            
            figure(f2)
            set(gcf,'name', sprintf('  Neuron %d, Trial %d _ %s _ %d Hz, %s', ineu, itr, labels_name{ismember(labels, alldata(itr).outcome)}, stimrate(itr), alldata(itr).correctSideName))
            %         set(gcf, 'name', sprintf('  Trial %d, bl %.2f, bl_th  %.2f , len_loHi: %d %d', itr, blPrecedEvent(ineu,itr), bl_th, len_loHi))
            %         set(gcf, 'name', sprintf('  Trial %d, baseline %.2f  %.2f  %.2f', itr, blPrecedEvent(ineu,itr), bl_th, bl_stable))
            hold on
            
            %%
            %         toplot = alldataDfofGood_fut(:,ineu,itr);
            %         t = 1:length(toplot);
            %         plot(toplot)
            
            timev = alldata(itr).frameTimes;
            
            % plot manual dfof
            if compareManual
                warning('you need to provide alldatadfofgood for manual trace. It is risky to use csfrs to derive that in case some imaging trials are missing.')
                toplot2 = dFOF_man(csfrs(itr)+1:csfrs(itr+1), imatched);
                if addNaNiti % bc the last frame will be NaN.
                    toplot2 = toplot2(1:end-1);
                end
                plot(timev, toplot2)
                hold on
            end
            
            % plot Eft dfof
            %             for itp = 1:length(traceFU_toplot)
            %                 toplot = traceFU_toplot{itp}(csfrs(itr)+1:csfrs(itr+1), ineu); % alldataDfofGood{itr}(:,ineu);  % check these 2 are identical if traceFU_toplot{1} is dfof.
            %                 if addNaNiti % bc the last frame will be NaN.
            %                     toplot = toplot(1:end-1);
            %                 end
            toplot = alldataDfofGood{itr}(:,ineu)';
            plot(timev, toplot)
            %             end
            
            % plot spikes
            if showSpikes
                mx = max(toplot(:));
                %                 toplot3 = snan(csfrs(itr)+1:csfrs(itr+1), ineu);
                %                 if addNaNiti % bc the last frame will be NaN.
                %                     toplot3 = toplot3(1:end-1);
                %                 end
                toplot3 = snantr{itr}(:, ineu);
                plot(timev, toplot3*mx, 'k.', 'markersize', 10);
            end
            
            
            %% plot wheel revolution.
            wheelTimeRes = alldata(itr).wheelSampleInt;
            wheelTimes = wheelTimeRes / 2 +  wheelTimeRes * (0:length(alldata(itr).wheelRev)-1);
            wheelRev = alldata(itr).wheelRev;
            wheelRev = wheelRev - wheelRev(1); % since the absolute values don't matter, I assume the position of the wheel at the start of a trial was 0. I think values are negative, bc when the mouse moves forward, rotary turns counter clock wise.
            
            plot(wheelTimes, wheelRev, 'g-.')
            %             plot(wheelTimes, -wheelRev, 'b')
            
            %% mark initTone:  plot lines at time points where specific events happen
            mn = 0; % min(toplot(:));
            %             mn = min([toplot(:); wheelRev]);
            mx = 1; % max(toplot(:));
            
            if markEventTimes
                plotEventTimes
            end
            %{
            f = ceil(timeInitTone{itr}(1)/frameLength);
            mn = min(toplot(:));
            mx = max(toplot(:));
            plot([f f], [mn mx], 'k:')
            %}
            
            %% mark bl_th and blDfof
            plot([timev(1) timev(end)], [bl_th bl_th], 'r:')
            
            if blPrecedEvent(ineu,itr) > bl_th
                plot(2,bl_th,'r*')
            end
            
            plot([timev(1) timev(end)], [blDfof(ineu) blDfof(ineu)], 'r:')
            
            %%
%             xlim([timev(1) timev(end)])
            xlim([wheelTimes(1) wheelTimes(end)])
            %             ylim([min(toplot(:))-.02  max(bl_th, max(toplot(:)))+.02])
            %             ylim([min([toplot(:); wheelRev])-.02  max(bl_th, max(toplot(:)))+.02])
            set(gca,'tickdir','out')
            %         xlabel('Frames')
            xlabel('Time from bcontrol trialStart (ms)')
            ylabel('DF/F')
            
            %%
            pause
            figure(f2)
            delete(gca)
            
        end
    end
    % N2, T20, B1.9, a big response later in the trial.
    
    
    %%
    pause
    
    figure(f1)
    if showitiFrNums, 
%         delete(t), 
    end
    cla
    %     delete(h)
    %     set(gca, 'tickdir','out', 'NextPlot', 'replacechildren');
    
    end
    
end






%% set to NaN trials with unstable baselines.
%{
bl_th = 1.5; % bl_stable*5; % *5 needs assessment.
traceEventAlign_stableBl = alldataDfofGood_fut;
traceEventAlign_stableBl(:, blPrecedEvent>bl_th) = NaN;


%%
% now average all trs for each neuron

traceAveTrs = nanmean(traceEventAlign_stableBl,3);
figure;
for ineu = 1:size(traceAveTrs,2)
    %     hold on
    plot(traceAveTrs(:,ineu))
    %     plot([f f], [mn mx], 'k:')
    
    pause
end
%}

