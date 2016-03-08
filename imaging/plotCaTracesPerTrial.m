% set some params related to behavior and trial types.
behavior_info

%%
plotNeurons1by1 = 0;

labels = [1 0 -1 -2 -3 -4 -5];
labels_name = {'success', 'failure', 'early decision', 'no decision', 'wrong start', 'no center commit', 'no side commit'};

wheelTimeRes = alldata(1).wheelSampleInt;


%%
figure; set(gca,'tickdir','out')

for itr = 1 : length(alldataDfofGood) % length(traceOriginalGood)
    
    hold on
    %     toplot = traceOriginalGood{itr};
    toplot = alldataDfofGood{itr}; % frames x units
    
    set(gcf,'name', sprintf('  Trial%s _ %s _ %d Hz, %s', num2str(itr), labels_name{ismember(labels, alldata(itr).outcome)}, stimrate(itr), alldata(itr).correctSideName))
    
    hns = plot(alldata(itr).frameTimes, toplot, 'color', [.6 .6 .6]); %  alldata.frameTimes is relative to bcontrol scope ttl time.
    ha = plot(alldata(itr).frameTimes, nanmean(toplot,2), 'k');
    
    xlim([alldata(itr).frameTimes(1)  alldata(itr).frameTimes(end)])
    %     xlabel('Time from the beginning of start\_rotary\_scope (ms) ',  'Interpreter', 'tex')
    xlabel('Time from bcontrol trialStart (ms)')
    ylabel('DF/F')
    
    
    %% plot lines at time points where specific events happen
    mn = min(toplot(:));
    mx = max(toplot(:));
    
    plotEventTimes
    
    
    %% plot wheel revolution.
    wheelTimes = wheelTimeRes / 2 +  wheelTimeRes * (0:length(alldata(itr).wheelRev)-1);
    wheelRev = alldata(itr).wheelRev;
    wheelRev = wheelRev - wheelRev(1); % since the absolute values don't matter, I assume the position of the wheel at the start of a trial was 0. I think values are negative, bc when the mouse moves forward, rotary turns counter clock wise.
    
    plot(wheelTimes, wheelRev, 'b')
    
    
    %%
    pause
    delete(hns)
    delete(ha)
    
    %%
    if plotNeurons1by1
        for ineuron = 1:size(toplot,2)
            %             set(ha, 'color', 'y') % turn the average trace gray
            toplot2 = toplot(:,ineuron);
            h1n = plot(alldata(itr).frameTimes, toplot2, 'color', [.7 .7 .7]);
            %             h1ns = plot(alldata(itr).frameTimes, smooth(toplot2,6), 'k');
            
            %             mn = min(toplot2(:));
            mx = max(toplot2(:));
            ylim([mn-.03 mx+.2])
            
            pause
            delete(h1n)
            %             delete([h1n, h1ns])
        end
    end
    
    %%
    delete(gca)
    
end

% look at the traces of each neuron one by one.
% concatenate all trials and look at their traces.
% look at ca transients... do they make sense?
% see if you need to normalize the activity of each neuron.
% to see how much of the response is due to wheel motion, look at response
% averages for high and low wheel motion.





