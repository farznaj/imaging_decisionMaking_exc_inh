%% set ca mean
blWinMs = [-97 -1];
blWin = eventFrameNum + (floor(blWinMs(1)/frameLength) : floor(blWinMs(2)/frameLength)); % eventFrameNum-3:eventFrameNum % 3 frames preced event.

% ca mean on the avage-trial trace
blPrecedEvent = nanmean(aveTrsPerNeuron(blWin,:)); % 1 x units, ca mean within blWin for each neuron
caMeanInRespWin = nanmean(aveTrsPerNeuron(respWin,:)); % 1 x units, ca mean within respWin for each neuron

%{
caPeakInRespWin = NaN(1, size(aveTrsPerNeuron,2));
for in = 1:size(aveTrsPerNeuron,2)
    [pks,~, ~, ~] = findpeaks(aveTrsPerNeuron(respWin,in)); %'minpeakheight', blDfof(in)+.05  'minpeakdistance', 3, 'threshold', .005);
    if ~isempty(pks)
        caPeakInRespWin(in) = max(pks);
    end
end
    
figure; plot([blPrecedEvent; caMeanInRespWin]')
figure; plot([blPrecedEvent; caPeakInRespWin]')

ratio_caPeak_bl = caPeakInRespWin ./ blPrecedEvent;
figure; plot(ratio_caPeak_bl)
[s,i] = sort(ratio_caPeak_bl,'descend');
i = [i(~isnan(s))  i(isnan(s))];
%}


% ca mean for each trial trace
caMeanInRespWin_eachTr = squeeze(nanmean(traceEventAlign(respWin,:,:)))'; % trials x units, ca mean within respWin for each trial and each neuron
blPrecedEvent_eachTr = squeeze(nanmean(traceEventAlign(blWin,:,:)))'; % trials x units, ca mean within blWin for each trial and each neuron


%% ttest2 for ca evoked and bl.
[h, p] = ttest2(blPrecedEvent_eachTr, caMeanInRespWin_eachTr); % , 'tail', 'left');
find(h)
[s_ttestp, i_ttestp] = sort(p);

figure; hold on;
plot(p)
plot([0 length(h)],[.05 .05])


%% plot ca ave for each trial and compare with bl.
% ave = [nanmean(blPrecedEvent_eachTr), nanmean(caMeanInRespWin_eachTr)];
% figure; plot(ave)

aveevok = nanmean(caMeanInRespWin_eachTr);

avebl = nanmean(blPrecedEvent_eachTr);
sdbl = nanstd(blPrecedEvent_eachTr); % ,0,1);
ntrs = sum(~isnan(blPrecedEvent_eachTr));
sebl = sdbl ./ sqrt(ntrs);
b = sebl; % sebl;
x = 1:length(avebl);

figure; 
boundedline(x, avebl, b)
plot(x, aveevok, 'r')
% plot(x, h)


%% set hists of caEvok and bl for each neuron
nbins = 20;
nunits = size(traceEventAlign,2);
n_evok = NaN(nbins-1, nunits);
n_bl = NaN(nbins-1, nunits);
edges = NaN(nbins, nunits);

for in = 1:nunits
    
    caEvok = caMeanInRespWin_eachTr(:, in);
    caBl = blPrecedEvent_eachTr(:, in);
    edges(:,in) = linspace(min([caEvok; caBl]), max([caEvok; caBl]), nbins);
    
    n_evok(:,in) =  histcounts(caEvok, edges(:,in), 'normalization', 'probability');
    n_bl(:,in) =  histcounts(caBl, edges(:,in), 'normalization', 'probability');
    
end


%% compare hists of ca evoked and bl for each neuron
figure; 
for in = randperm(nunits), % 1:nunits
    x = edges(1:end-1,in) + mode(diff(edges(:,in)))/2;
    y = [n_bl(:,in), n_evok(:,in)];
%     y_filtered = boxFilter(y, 3, 1, 0);
    
    plot(x, y)
%     plot(x, y_filtered)

%     xlim([min(edges(:)) max(edges(:))])
    ylim([-.05 1])
    set(gca,'tickdir', 'out')
    set(gcf, 'name', num2str(in))
    box off
    
    pause
end


%%














