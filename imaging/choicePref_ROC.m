function choicePref_all = choicePref_ROC(traces_al_sm, ipsiTrs, contraTrs, makeplots, eventI_stimOn, useEqualNumTrs)
% choicePref_all: frames x neurons
% compute choice preference (2*(auc-0.5)) for each neuron at each frame.
% use avetrialAlign_setVars to get the input vars.

%%
if ~exist('makeplots', 'var')
    makeplots = false;
end

if ~exist('useEqualNumTrs', 'var')
    useEqualNumTrs = false; % if true, equal number of trials for HR and LR will be used to compute ROC.
end


%% If desired, use equal number of trials for both HR and LR conditions.
%  You don't have to do this.

if useEqualNumTrs
    cond1 = find(ipsiTrs);
    cond2 = find(contraTrs);
    % fprintf('N trials for ipsi and contra = %d  %d\n', [sum(ipsiTrs), sum(contraTrs)])
    
    extraTrs = setRandExtraTrs(cond1, cond2);
    
    % make sure ipsi and contra have equal number of trials.
    ipsiTrs(extraTrs) = 0;
    contraTrs(extraTrs) = 0;
    
    fprintf('N trials for ipsi and contra = %d  %d\n', [sum(ipsiTrs), sum(contraTrs)])
end

if ~sum(ipsiTrs) || ~sum(contraTrs) % abort the analysis if one of the two conditions doesn't have any trials.
    error('One of the conditions has 0 trials!')
    %     choicePref_all = [];
    %     return
end

% critList = unique(traces_al_sm);
% critList(isnan(critList)) = [];


%% Compute choice preference for each neuron at each time point during a trial (only correct trials)

% if you are using matlab roc, make sure there are no NaNs in the outputs
% (traces), otherwise it wont compute trp and fpr accurately!

numfrs = size(traces_al_sm, 1);
choicePref_all = NaN(numfrs, size(traces_al_sm, 2));
targets = [zeros(numfrs, sum(ipsiTrs)), ones(numfrs, sum(contraTrs))];

for in = 1:size(traces_al_sm, 2)
    
    %%
    if numfrs > 1
        traces_i = squeeze(traces_al_sm(:, in, ipsiTrs)); % frames x trials.
        traces_c = squeeze(traces_al_sm(:, in, contraTrs)); % frames x trials.
    else
        traces_i = squeeze(traces_al_sm(:, in, ipsiTrs))'; % frames x trials.
        traces_c = squeeze(traces_al_sm(:, in, contraTrs))'; % frames x trials.
    end
    
    %
    outputs = [traces_i, traces_c]; % ipsi is asigned 0 and contra is asigned 1 in "targets". so assumption is ipsi response is lower than contra. so auc>.5 (choicePref>0) happens when ipsi resp<contra, and auc<.5 (choicePref<0) happens when ipsi>contra.
    if all(isnan(outputs(:)))
        fprintf('The neural trace is all NaNs!\n')
    else
        
        if any(isnan(outputs(:))), error('There are NaNs in outputs (traces). Matlab roc wont work properly. Get rid of NaNs!'), end
        
        [tpr,fpr,thresholds] = roc(targets,outputs); % each cell corresponds to a frame
        
        %     tpr = cellfun(@(x)smooth(x,5)', tpr, 'uniformoutput', 0); % smoothing makes very little difference
        %     fpr = cellfun(@(x)smooth(x,5)', fpr, 'uniformoutput', 0);
        %{
    figure; plotroc(targets(fr,:), outputs(fr,:))
    figure; plotroc(targets, outputs)
        %}
        
        if numfrs > 1
            % auc = trapz([0, fpr{fr}, 1], [0, tpr{fr}, 1]); % choice pref measure for each neuron at each frame
            auc = cellfun(@(x,y)trapz([0, x, 1], [0, y, 1]), fpr, tpr); % [fpr, tpr] = [0 0] and [1 1] will be always valid and you need them to measure auc correctly.
        else
            auc = trapz([0, fpr, 1], [0, tpr, 1]);
        end
        
        choicePref =  2*(auc-0.5);
        %     figure; plot(choicePref) % look at choicePref for neuron in over time (all frames)
        choicePref_all(:,in) = choicePref; % frames x neurons
        
        %     figure; hold on
        %     boundedline((1:numfrs)', nanmean(traces_i,2), nanstd(traces_i,[],2), 'alpha')
        %     boundedline((1:numfrs)', nanmean(traces_c,2), nanstd(traces_c,[],2), 'r', 'alpha')
    end
    
    
end
fprintf('Size of choicePref_all (fr x units): %d  %d\n', size(choicePref_all))

% choicePref_all(:, ~good_inhibit) % excitatory neurons.
% choicePref_all(:, good_inhibit) % inhibitory neurons.


%% Make some plots

% some neurons seem to have choice preference before the stim is presented...

if makeplots
    
    figure;
    % plot average choicePref across neurons
    subplot(221), hold on
    top = nanmean(abs(choicePref_all),2);
    ylabel('Abs(choicePref)')
    xlabel('Frames')
%     top = nanmean((choicePref_all),2);
    plot(top, 'k', 'linewidth', 2)
    plot([eventI_stimOn  eventI_stimOn], [min(top(:))  max(top(:))], 'k--')
    % figure; plot(nanmean(choicePref_all(:,~good_inhibit),2))
    
    % compare hist of choice pref in the 1st and 2nd half of the stim.
    frsh = round(linspace(eventI_stimOn, numfrs, 3));
    choicePref_all_h1 = nanmean(choicePref_all(frsh(1):frsh(2), :)); % average choicePref during 1st half of the stimulus
    choicePref_all_h2 = nanmean(choicePref_all(frsh(end-1):frsh(end), :)); % average choicePref during 2nd half of the stimulus
    fprintf('mean half1 and half2 of choicePref: %.3f  %.3f\n', nanmean(choicePref_all_h1), nanmean(choicePref_all_h2));
    
    v = linspace(min(choicePref_all(:)), max(choicePref_all(:)), 10);
    n1 = histcounts(choicePref_all_h1, v, 'normalization', 'count');
    n2 = histcounts(choicePref_all_h2, v, 'normalization', 'count');
    
    box off
    set(gca,'tickdir','out')
    set(gca,'ticklength',[.025 0])
    
    
    subplot(224)
    plot(v(1:end-1), [n1; n2]')
    legend('half 1','half 2')
    
    
    
    % plot histogram of chiocePref for the first few frames and the last
    % few frames.
    subplot(222), hold on; 
%     plot(v(1:end-1), cumsum([n1; n2]'))
    b = choicePref_all(size(choicePref_all,1) - 2 : size(choicePref_all,1),:);
    histogram(b(:), 'normalization', 'probability')
    a = choicePref_all(eventI_stimOn-2 : eventI_stimOn,:);
    histogram(a(:), 'normalization', 'probability');
    legend('End of stim','Pre-stim')
    legend boxoff
    xlabel('ipsi <-- Choice pref --> contra')
    ylabel('Fraction')
    
    box off
    set(gca,'tickdir','out')
    set(gca,'ticklength',[.025 0])
    
    
    
    % plot choice pref for all neurons for each frame
    a = choicePref_all';
    subplot(223), hold on
    imagesc(a)
    plot([eventI_stimOn  eventI_stimOn], [1  size(traces_al_sm, 2)], 'r')
    axis tight
    h = colorbar;
    h.Label.String = 'ipsi <-- choice pref --> contra';
    xlabel('Frames')
    ylabel('Neurons')
    % s = sortrows(a, numfrs);
    % figure; imagesc(s)
    
end


%% manual method: compute false-positive and true-positive for ROC
% you have confirmed that the manual method (codes below) and matlab roc
% give the same result. matlab roc is much faster though.
%{
% ipsi chioce (left)
critList = unique(traces_al_sm);
critList(isnan(critList)) = [];


% for each neuron, compute TP and FP at each frame across all trials.
% TP: number of left trials with signal > th (at frame f) divided by
% total number of left trials.
% FP: number of right trials with signal > th (at frame f) divided by total
% number of right trials.

tp = NaN(numfrs, length(critList));
fp = NaN(numfrs, length(critList));
for ith = 1:length(critList)
    th = critList(ith);
    tp(:,ith) = sum(traces_c>th, 2) / size(traces_c, 2); % frames x 1. tp for each frame.
    fp(:,ith) = sum(traces_i>th, 2) / size(traces_i, 2); % frames x 1. fp for each frame.
end

auc = NaN(1, numfrs);
for fr = 1:numfrs
    auc(fr) = trapz(sort([1, fp(fr,:), 0]), sort([1, tp(fr,:), 0]));
end

%%%%
% manual
figure; hold on
plot([0 1],[0 1])
plot([1, fp(fr,:), 0], [1, tp(fr,:), 0])

% matlab roc
figure; hold on
plot([0 1],[0 1])
plot([0, fpr{fr}, 1], [0, tpr{fr}, 1])


% plot(nanmean(fp), nanmean(tp)) % across all frames does the neuron respond more to ipsi or contra stimulus.
%}

