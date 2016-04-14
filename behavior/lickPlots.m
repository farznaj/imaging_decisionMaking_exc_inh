% Look at licks for each trial one by one and plot a distribution of lick
% durations.

% script lickAlign aligns licks on trial events and averages across trials.


%% Look at licks and mark event times for each trial

figure('position', [440   508   864   290]);

for itr = 1:length(alldata)
    hold on,
    set(gcf,'name', sprintf('trial %d, outcome %d ', itr, alldata(itr).outcome))
    xlabel('Time since scopeTTL (ms)')
    
    plotEventTimes
    
    pause
    clf
end


%% Plot histogram of duration of licks (on to off length in the pokes field).

centLickDur =  cell(1, length(alldata));
leftLickDur =  cell(1, length(alldata));
rightLickDur =  cell(1, length(alldata));
for tr = 1:length(alldata)
    centLickDur{tr} = diff(alldata(tr).parsedEvents.pokes.C, [] ,2);
    leftLickDur{tr} = diff(alldata(tr).parsedEvents.pokes.L, [] ,2);
    rightLickDur{tr} = diff(alldata(tr).parsedEvents.pokes.R, [] ,2);
    %     centLickDur{tr} = alldata(tr).parsedEvents.pokes.C(:,2);
    %     leftLickDur{tr} = alldata(tr).parsedEvents.pokes.L(:,2);
    %     rightLickDur{tr} = alldata(tr).parsedEvents.pokes.R(:,2);
end

allC = cell2mat(centLickDur(:));
allL = cell2mat(leftLickDur(:));
allR = cell2mat(rightLickDur(:));
C_ave_min_max = [nanmean((allC))  min(allC)  max(allC)]
L_ave_min_max = [nanmean((allL))  min(allL)  max(allL)]
R_ave_min_max = [nanmean((allR))  min(allR)  max(allR)]
a = [C_ave_min_max;L_ave_min_max;R_ave_min_max];
ed = linspace(min(a(:,2)), max(a(:,3)), 100);
% [m,i] = max(allL)
% a = cellfun(@length, leftLickDur);
% figure; plot(cumsum(a))

[n1,ed1] = histcounts(allC, ed, 'normalization', 'probability');
[n2,ed2] = histcounts(allL, ed, 'normalization', 'probability');
[n3,ed3] = histcounts(allR, ed, 'normalization', 'probability');

figure;

subplot 211, hold on
plot(ed(1:end-1), n1)
plot(ed(1:end-1), n2)
plot(ed(1:end-1), n3)
legend('C','L','R')

subplot 212, hold on
plot(ed(1:end-1), cumsum(n1))
plot(ed(1:end-1), cumsum(n2))
plot(ed(1:end-1), cumsum(n3))

