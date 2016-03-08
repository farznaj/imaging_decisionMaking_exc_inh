% heatmaps

%% Find event indeces on the concatenated trace.

fr_len_traces_all = [size(traces_aligned_fut_initTone, 1), ...
    size(traces_aligned_fut_stimOn, 1), ...
    size(traces_aligned_fut_goTone, 1), ...
    size(traces_aligned_fut_1stSideTry, 1), ...
    size(traces_aligned_fut_reward, 1)];

cs_traces_o = [0 cumsum(fr_len_traces_all)];
beg_each_trace = 1+cs_traces_o(1:end-1);
beg_each_trace_aftNaN = beg_each_trace + [0:length(traces_aligned_all)-1]; % 4 NaNs were inserted.

eventI_all = [eventI_initTone eventI_stimOn eventI_goTone eventI_1stSideTry eventI_reward];

eventI_catTrace = beg_each_trace_aftNaN + (eventI_all-1); % index of events on the concatenated trace.


%% single trials, sorted by when max response happened.

%% correct left choice trials
figure('name', 'Corect left-choice trials');
ha = tight_subplot(nr,nc,[.03 .02],[.03 .001],[.03 .001]);

for itr = 1:size(traces_aligned_corrL,3)
    
    trace_allN_1tr_cl = traces_aligned_corrL(:,:,itr);
    
    %% sort neurons based on their frame index of max signal.
    % for each neuron find the frame (i) and the magnitude (m) of max signal.
    [m, i] = max(trace_allN_1tr_cl); % i is the frame index of max signal for each trace
    
    % throw away neurons whose max is 0.
    trace_nonZN_1tr_cl = trace_allN_1tr_cl(:, m~=0); % neurons that has at least some rise in their trace. m~=0 can be used if we are working w spikes not C (temporal component).
    mnew = m(m~=0);
    inew = i(m~=0);
    
    % sort neurons based on their frame index of max signal.
    [sfr, isfr] = sort(inew); % sfr is the frame index of max signal for each trace (sorted ascendignly). isfr is the trace (unit) index.
    
    %%
    hold(ha(itr), 'on')
    imagesc(trace_nonZN_1tr_cl(:, isfr)', 'parent', ha(itr))
%     freezeColors
    plot(ha(itr), [eventI_catTrace([1,3,5]); eventI_catTrace([1,3,5])], [1 size(trace_nonZN_1tr_cl, 2)], 'r:')
    plot(ha(itr), [eventI_catTrace([2,4]); eventI_catTrace([2,4])], [1 size(trace_nonZN_1tr_cl, 2)], 'r')
    
    axis(ha(itr), 'tight')
    set(gca, 'ydir', 'normal')
    
%     pause
    
end
ylabel(ha(1), 'Neurons')
xlabel(ha(1), 'Time')



%% correct right choice trials
figure('name', 'Corect right-choice trials');
ha = tight_subplot(nr,nc,[.03 .02],[.03 .001],[.03 .001]);

for itr = 1:size(traces_aligned_corrR,3)
    
    trace_allN_1tr_cr = traces_aligned_corrR(:,:,itr);
    
    %% sort neurons based on their frame index of max signal.
    % for each neuron find the frame (i) and the magnitude (m) of max signal.
    [m, i] = max(trace_allN_1tr_cr); % i is the frame index of max signal for each trace
    
    % throw away neurons whose max is 0.
    trace_nonZN_1tr_cr = trace_allN_1tr_cr(:, m~=0); % neurons that has at least some rise in their trace. m~=0 can be used if we are working w spikes not C (temporal component).
    mnew = m(m~=0);
    inew = i(m~=0);
    
    % sort neurons based on their frame index of max signal.
    [sfr, isfr] = sort(inew); % sfr is the frame index of max signal for each trace (sorted ascendignly). isfr is the trace (unit) index.
    
    %%
    hold(ha(itr), 'on')
    imagesc(trace_nonZN_1tr_cr(:, isfr)', 'parent', ha(itr))
%     freezeColors
    plot(ha(itr), [eventI_catTrace([1,3,5]); eventI_catTrace([1,3,5])],[1 size(trace_nonZN_1tr_cr, 2)], 'r:')
    plot(ha(itr), [eventI_catTrace([2,4]); eventI_catTrace([2,4])],[1 size(trace_nonZN_1tr_cr, 2)], 'r')
    
    axis(ha(itr), 'tight')
    set(gca, 'ydir', 'normal')
    
%     pause
    
end
ylabel(ha(1), 'Neurons')
xlabel(ha(1), 'Time')



%% sort neurons based on their max response on the trial averaged trace
tro = traces_aligned_corrL;
trace = nanmean(tro,3);
size(trace)

[m, ~] = max(trace); % max of each neuron across all frames
[~, i] = sort(m); % sorting neurons based on their max response on the trial averaged trace
figure; imagesc(trace(:,i)')
set(gca, 'ydir', 'normal')


%% look at single trials sorted by their max response on the trial-averaged trace
figure('name', 'Corect left-choice trials');
ha = tight_subplot(nr,nc,[.03 .02],[.03 .001],[.03 .001]);

for itr = 1:size(traces_aligned_corrR,3)
    trace_allN_1tr_cl = traces_aligned_corrR(:,:,itr);
    trace_nonZN_1tr_cl = trace_allN_1tr_cl;

    hold(ha(itr), 'on')
    imagesc(trace_nonZN_1tr_cl(:, i)', 'parent', ha(itr))
%     freezeColors
    plot(ha(itr), [eventI_catTrace([1,3,5]); eventI_catTrace([1,3,5])], [1 size(trace_nonZN_1tr_cl, 2)], 'r:')
    plot(ha(itr), [eventI_catTrace([2,4]); eventI_catTrace([2,4])], [1 size(trace_nonZN_1tr_cl, 2)], 'r')
    
    axis(ha(itr), 'tight')
    set(gca, 'ydir', 'normal')
    
%     pause
    
end
ylabel(ha(1), 'Neurons')
xlabel(ha(1), 'Time')


%% trial-averaged traces for all neurons for left and right correctchoices.
% tro = traces_aligned_corrR;
% trace = nanmean(tro,3);
% size(tro), size(trace)

figure; 
subplot(121)
tro = traces_aligned_corrL;
trace = nanmean(tro,3);
imagesc(trace(:,i)')
hold on
plot([eventI_catTrace([1,3,5]); eventI_catTrace([1,3,5])], [1 size(trace_nonZN_1tr_cl, 2)], 'r:')
plot([eventI_catTrace([2,4]); eventI_catTrace([2,4])], [1 size(trace_nonZN_1tr_cl, 2)], 'r')

subplot(122)
tro = traces_aligned_corrR;
trace = nanmean(tro,3);
imagesc(trace(:,i)')
hold on
plot([eventI_catTrace([1,3,5]); eventI_catTrace([1,3,5])], [1 size(trace_nonZN_1tr_cl, 2)], 'r:')
plot([eventI_catTrace([2,4]); eventI_catTrace([2,4])], [1 size(trace_nonZN_1tr_cl, 2)], 'r')


%%







%%
%{
%%
itr = 1;
trace_allN_1tr_cl = traces_aligned_corrL(:,:,itr);
% trace_allN_1tr_cl = traces_aligned_corrR(:,:,itr);

%%
figure; 
subplot(211)
hold on
imagesc(trace_allN_1tr_cl')
freezeColors
axis tight
plot([eventI_catTrace; eventI_catTrace]',[1 numGoodUnits], 'r')

%% sort neurons based on their frame index of max signal.
% for each neuron find the frame (i) and the magnitude (m) of max signal.
[m, i] = max(trace_allN_1tr_cl); % i is the frame index of max signal for each trace
% [i; m]'

% sort neurons based on their max signal
% [s, iss] = sort(m); % iss is the unit number and s is its max signal. sorted ascendingly.
% [s; iss]'
% figure; 
% imagesc(trace_allN_1tr_cl(:, iss)') % traces in iss order (max signal sorted).

 
% throw away neurons whose max is 0.
trace_nonZN_1tr_cl = trace_allN_1tr_cl(:, m~=0); % neurons that has at least some rise in their trace. m~=0 can be used if we are working w spikes not C (temporal component).
mnew = m(m~=0);
inew = i(m~=0);

% sort neurons based on their frame index of max signal.
[sfr, isfr] = sort(inew); % sfr is the frame index of max signal for each trace (sorted ascendignly). isfr is the trace (unit) index.
% [sfr; isfr]'

%%
% plot neurons sorted based on their frame index of max signal
figure;
imagesc(trace_nonZN_1tr_cl(:, isfr)')
set(gca, 'ydir', 'normal')
hold on
plot([eventI_catTrace; eventI_catTrace]',[1 numGoodUnits], 'r')
%}
%%

    
    
    