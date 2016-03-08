% set some params related to behavior and trial types.
% behavior_info

%% specify what traces you want to plot
traces = alldataSpikesGood; % alldataSpikesGoodInh; % %  % traces to be aligned.


%%
traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.

defaultPrePostFrames = 2; % default value for nPre and postFrames, if their computed values < 1.
shiftTime = 0; % event of interest will be centered on time 0 which corresponds to interval [-frameLength/2  +frameLength/2]
scaleTime = frameLength;


%%
alignedEvent = 'initTone';
[traces_aligned_fut_initTone, time_aligned_initTone, eventI_initTone] = alignTraces_prePost_allCases...
    (alignedEvent, traces, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward);

%%
alignedEvent = 'stimOn';
[traces_aligned_fut_stimOn, time_aligned_stimOn, eventI_stimOn] = alignTraces_prePost_allCases...
    (alignedEvent, traces, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward);

%%
alignedEvent = 'goTone';
[traces_aligned_fut_goTone, time_aligned_goTone, eventI_goTone] = alignTraces_prePost_allCases...
    (alignedEvent, traces, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward);

%%
alignedEvent = '1stSideTry';
[traces_aligned_fut_1stSideTry, time_aligned_1stSideTry, eventI_1stSideTry] = alignTraces_prePost_allCases...
    (alignedEvent, traces, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward);

%%
alignedEvent = 'reward';
[traces_aligned_fut_reward, time_aligned_reward, eventI_reward] = alignTraces_prePost_allCases...
    (alignedEvent, traces, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward);


%%
traces_aligned_all = {'traces_aligned_fut_initTone', 'traces_aligned_fut_stimOn', ...
    'traces_aligned_fut_goTone', 'traces_aligned_fut_1stSideTry', 'traces_aligned_fut_reward'};

eventI_all = {'eventI_initTone', 'eventI_stimOn', 'eventI_goTone', 'eventI_1stSideTry', 'eventI_reward'};

time_all = {'time_aligned_initTone', 'time_aligned_stimOn', 'time_aligned_goTone', ...
    'time_aligned_1stSideTry', 'time_aligned_reward'};


%% concatenate the aligned traces with a NaN in between
nu = size(traces{1},2);
traces_aligned_cat = cat(1, ...
    traces_aligned_fut_initTone, NaN(1, nu, numTrials), ...
    traces_aligned_fut_stimOn, NaN(1, nu, numTrials), ...
    traces_aligned_fut_goTone, NaN(1, nu, numTrials), ...
    traces_aligned_fut_1stSideTry, NaN(1, nu, numTrials), ...
    traces_aligned_fut_reward);

fprintf('size of traces_aligned_cat: %d  %d  %d\n', size(traces_aligned_cat))


%%%
fprintf('Num correct trials for Left and Right side: %d  %d\n', [sum(correctL), sum(correctR)])

traces_aligned_corrL = traces_aligned_cat(:,:,correctL);
traces_aligned_corrR = traces_aligned_cat(:,:,correctR);

aveTrsPerNeuron_corrL = nanmean(traces_aligned_corrL, 3);
aveTrsPerNeuron_corrR = nanmean(traces_aligned_corrR, 3);

sdTrsPerNeuron_corrL = nanstd(traces_aligned_corrL, 0, 3);
seTrsPerNeuron_corrL = sdTrsPerNeuron_corrL ./ ...
    repmat(sqrt(sum(correctL)), size(sdTrsPerNeuron_corrL));

sdTrsPerNeuron_corrR = nanstd(traces_aligned_corrR, 0, 3);
seTrsPerNeuron_corrR = sdTrsPerNeuron_corrR ./ ...
    repmat(sqrt(sum(correctR)), size(sdTrsPerNeuron_corrR));


%%% compute min and max of ave+se traces of all neurons (u may need it for plotting)
mnL = min(aveTrsPerNeuron_corrL - seTrsPerNeuron_corrL);
mnR = min(aveTrsPerNeuron_corrR - seTrsPerNeuron_corrR);
mn = min([mnL;mnR]);

mxL = max(aveTrsPerNeuron_corrL + seTrsPerNeuron_corrL);
mxR = max(aveTrsPerNeuron_corrR + seTrsPerNeuron_corrR);
mx = max([mxL;mxR]);


%% set gaussian filter
dofilter = true;

if dofilter
    siz = 5;
    sig = 1;
    x = -floor(siz/2) : floor(siz/2);
    H = exp(-(x.^2/ (2 * sig^2)));
    H = H/sum(H);
end


%% plot trigger aligned traces

nr = 7; nc = 7;
figall = [0: nr*nc: nu , nu];

for ifig = 1:length(figall)-1
    
    figure('name', sprintf('Neurons %d-%d', figall(ifig)+1, figall(ifig+1)));
    ha = tight_subplot(nr,nc,[.03 .02],[.03 .001],[.03 .001]);
    set(ha, 'Yticklabelmode', 'auto')
    cnt = 0;
    
    %%
    for ineu = figall(ifig)+1 : figall(ifig+1) % 1:size(traceEventAlign,2) % randperm(size(traceEventAlign,2)) % i_ttestp; %  %
        cnt = cnt+1;
        axes(ha(cnt));
%         title(ineu)
        hold on
        st = 1; gp = 2;
        ttot = []; eltot = [];
        mnx = [mn(ineu) mx(ineu)]; % [-.01 .015];
        
        %%
        for isec = 1:length(traces_aligned_all) % loop over traces aligned on different events
            
            tp = eval(traces_aligned_all{isec}); % traces_aligned_fut_initTone;
            el = eval(eventI_all{isec}); % eventI_initTone;
            tt = eval(time_all{isec});
            
            ln = size(tp, 1);
            xsec = st : st+ln-1;
            el = xsec(1)+el-1;
            st = xsec(end)+gp+1;
            ttot = [ttot, tt, NaN(1,gp)];
            eltot = [eltot, el];
            
            traces_aligned_corrL = tp(:,ineu,correctL);
            traces_aligned_corrR = tp(:,ineu,correctR);
            
            aveTrsPerNeuron_corrL = nanmean(traces_aligned_corrL, 3);
            aveTrsPerNeuron_corrR = nanmean(traces_aligned_corrR, 3);
            
            sdTrsPerNeuron_corrL = nanstd(traces_aligned_corrL, 0, 3);
            seTrsPerNeuron_corrL = sdTrsPerNeuron_corrL ./ ...
                repmat(sqrt(sum(correctL)), size(sdTrsPerNeuron_corrL));
            
            sdTrsPerNeuron_corrR = nanstd(traces_aligned_corrR, 0, 3);
            seTrsPerNeuron_corrR = sdTrsPerNeuron_corrR ./ ...
                repmat(sqrt(sum(correctR)), size(sdTrsPerNeuron_corrR));
            
            
            %% left choice (ipsi): blue
            toplot = aveTrsPerNeuron_corrL; %(:, ineu);
            toplot_b = seTrsPerNeuron_corrL; %(:, ineu);
            
            if dofilter
                toplot = conv(toplot, H', 'same');
                toplot_b = conv(toplot_b, H, 'same');
                
                % you need to use the following if you don't use the 'same' option for conv
%                 toplot = toplot(ceil(siz/2) : end-ceil(siz/2)+1);
%                 toplot_b = toplot_b(ceil(siz/2) : end-ceil(siz/2)+1);
            end
            
            %     plot(t, toplot, 'color', [114 189 255]/256)
            %     plot(t, toplot+toplot_sd, ':', 'color', [114 189 255]/256)
            %     plot(t, toplot-toplot_sd, ':', 'color', [114 189 255]/256)
            
            boundedline(xsec, toplot, toplot_b, 'alpha');
            %         set(h, 'linewidth', 2)
            
            %% right choice (contra) : red
            toplot = aveTrsPerNeuron_corrR; %(:, ineu);
            toplot_b = seTrsPerNeuron_corrR; %(:, ineu);
            
            if dofilter
                toplot = conv(toplot, H', 'same');
                toplot_b = conv(toplot_b, H, 'same');
                
                % you need to use the following if you don't use the 'same' option for conv                
%                 toplot = toplot(ceil(siz/2) : end-ceil(siz/2)+1);
%                 toplot_b = toplot_b(ceil(siz/2) : end-ceil(siz/2)+1);
            end
            
            boundedline(xsec, toplot, toplot_b, 'r', 'alpha')
            
            %% plot a line and write text for the event
            plot([el el], mnx, 'k:')
            
            tx = eventI_all{isec};
            %         text(el-2, mnx(1)-.002, tx(strfind(tx, '_')+1:end))
            text(el-2, mnx(2)-.002, tx(strfind(tx, '_')+1:end))
            
        end
        
        %%
        xlim([1 xsec(end)])
        ylim([mn(ineu)-.005  mx(ineu)+.005])
        %     xlim([-500 500])
        %     xlabel('Time since event onset (ms)')
        %     ylabel('DF/F')
        set(gca, 'tickdir', 'out')
        %     set(gcf, 'name', sprintf('Neuron %d', ineu))    %     set(gcf, 'name', sprintf('Neuron %d, P=%.2f', ineu, s_ttestp(cnt)))
        %     set(gca, 'xtick', eltot) % marks events
        %     set(gca, 'xtick', (1:7:length(ttot)))
        %     set(gca, 'xticklabel', round(ttot(1:7:end)))
        
        %     round(200/frameLength)
        e = [eltot(1)-6 eltot(1) eltot(2) eltot(2)+6 eltot(2)+12 eltot(2)+18 eltot(3) eltot(4) eltot(4)+6 eltot(5)-6 eltot(5) eltot(5)+6];
        set(gca, 'xtick', e)
        set(gca, 'xticklabel', round(ttot(e)))
        
        %%
%         pause
%         delete(gca)
        
    end
end



%% use all-events concatenated traces
%{
%% remember bc of the NaNs, filtering will give NaNs on the edges of each trace so don't do filtering for the _cat trace.
dofilter = 0;

figure;
cnt = 0;

for ineu = randperm(size(traceEventAlign,2)) % i_ttestp; %  % 1:size(traceEventAlign,2)
    cnt = cnt+1;
    hold on
    
    %%
    toplot = aveTrsPerNeuron_corrL(:, ineu);
%     toplot_b = sdTrsPerNeuron(frs2plot, ineu);
    toplot_b = seTrsPerNeuron_corrL(:, ineu);
    
    if dofilter
        toplot = conv(toplot, H');
        toplot_b = conv(toplot_b, H);
        
        toplot = toplot(ceil(siz/2) : end-ceil(siz/2)+1);
        toplot_b = toplot_b(ceil(siz/2) : end-ceil(siz/2)+1);
    end
    
%     plot(t, toplot, 'color', [114 189 255]/256)
%     plot(t, toplot+toplot_sd, ':', 'color', [114 189 255]/256)
%     plot(t, toplot-toplot_sd, ':', 'color', [114 189 255]/256)
    
    boundedline(1:length(toplot), toplot, toplot_b)
    
%     plot(t, aveTrsPerNeuron_S(frs2plot, ineu))


    %%
    toplot = aveTrsPerNeuron_corrR(:, ineu);
%     toplot_b = sdTrsPerNeuron(frs2plot, ineu);
    toplot_b = seTrsPerNeuron_corrR(:, ineu);
    
    if dofilter
        toplot = conv(toplot, H');
        toplot_b = conv(toplot_b, H);
        
        toplot = toplot(ceil(siz/2) : end-ceil(siz/2)+1);
        toplot_b = toplot_b(ceil(siz/2) : end-ceil(siz/2)+1);
    end
    
%     plot(t, toplot, 'color', [114 189 255]/256)
%     plot(t, toplot+toplot_sd, ':', 'color', [114 189 255]/256)
%     plot(t, toplot-toplot_sd, ':', 'color', [114 189 255]/256)
    
    boundedline(1:length(toplot), toplot, toplot_b, 'r')
    
    
    %% plot the average wheel movement across trials aligned on eventTime.
    if plotwheelRev
        plot(timeEventAlign_wheelRev(frs2plot_wheel), -aveTrsPerNeuron_wheel(frs2plot_wheel), 'k:')
    end
    
    
    %% put some markers
%     plot([1 length(toplot)], [blDfof(ineu)  blDfof(ineu)], 'r:')
    if plotmarkers
        % bl_th
        bl_th = quant80Peaks(ineu);
        plot([t(1) t(end)], [bl_th  bl_th], 'r:')
        

        % a line at 0 (ie the event time)
        plot([0 0], [min(toplot) max(toplot)])
    end
    
    %% response window
%     plot([respWinMs(1)  respWinMs(1)], [min(toplot-toplot_b) max(toplot+toplot_b)], 'g-.')
%     plot([respWinMs(2)  respWinMs(2)], [min(toplot-toplot_b) max(toplot+toplot_b)], 'g-.')
    
    
    %%
%     mn = min(bl_th, blDfof(ineu));
%     mx = max([bl_th; blDfof(ineu); toplot(:)]);
%     ylim([mn mx])
    
%     xlim([-500 500])
%     xlabel('Time since event onset (ms)')
    ylabel('DF/F')
    set(gca, 'tickdir', 'out')
    set(gcf, 'name', sprintf('Neuron %d', ineu))
%     set(gcf, 'name', sprintf('Neuron %d, P=%.2f', ineu, s_ttestp(cnt)))
    
    %%
    pause
    delete(gca)
    
end

%}
