function [HRchoicePerc, vec_rates, up, lo, nSamples, xx, yy, ee, hmodal] = PMF_set_plot(stimrate, allResp_HR_LR, cb, vec_rates, plotPMF, shownumtrs, lineColor, setWilsInt)
% set the x (vec_rates) and y (HRchoicePerc) for plotting the pshychometric function (PMF).
% also set the confidence intervals.
%
% allResp_HR_LR: vector 1 for HR, 0 for LR.
% vec_rates: optional.
%
% [~, ~, stimrate] = setStimRateType(alldata);
% outcomes = [alldata.outcome];
% outcomes(allowCorrectEntered) = 0; % their outcome can be 0, 1, or -5, but u're going with mouse's 1st choice.
% cb = unique([alldata.categoryBoundaryHz]); % categ boundary in Hz

if ~exist('plotPMF', 'var')
    plotPMF = true;
end

if ~exist('shownumtrs', 'var')
    shownumtrs = true;
end

if ~exist('setWilsInt', 'var')
    setWilsInt = true;
end


%% set vec_rates and bin_rates

th = 0; % 2; % min number of trials in a stim rate bin in order to compute %HR

if ~exist('vec_rates', 'var') || isempty(vec_rates)
    wd = 2;
    vec_rates = sort([cb : -wd : min(stimrate)-wd  ,  cb+wd : wd : max(stimrate)+wd]); % make sure categ boundary doesn't go at the middle of a bin!
else
    wd = mode(diff(vec_rates));
end

[~, bin_rate] = histc(stimrate, vec_rates);


%% compute prop HR for each bin of vec_rates

HRchoicePerc = NaN(1, length(vec_rates));
nSamples = NaN(1, length(vec_rates)); % number of trials in each bin of vec_rates

for ri = 1:length(vec_rates)
    
%     allout = outcomes(bin_rate == ri); % outcome of all trials whose stim rate was in bin ri
    allout = allResp_HR_LR(bin_rate == ri); % choice of mouse on all trials whose stim rate was in bin ri
    validout = allout(allout>=0); % only consider trials that were correct, incorrect, noDecision, and noSideLickAgain trials. (exclude non valid trials).
    nSamples(ri) = length(validout);
    
    if length(validout)>th
%         HRchoicePerc(ri) = nanmean(validout==1); % success rate
        HRchoicePerc(ri) = nanmean(validout); % success rate
    end
end

% using outcomes to compute HRchoicePerc and then the code below can work for all rates except
% when stim rate=cb, in which case knowing outcomes wont give us any info about
% the choice side.
% success rate for LR choices indicates percent LR. since we want to plot percent HR, then for LR choices it will be eqal to 1-success rate.
% HRchoicePerc(vec_rates < cb) = 1-HRchoicePerc(vec_rates < cb);



%% compute confidence bounds

% CI_alpha = 0.05; % alpha corresponding to CI=.95
% perc_normal = 1 - CI_alpha/2; % .975
% z = norminv(perc_normal, 0, 1); % 1.96; z corresponding to probability .975 in a standard normal distribution.

if setWilsInt
    z = 1; % 1 SEM

    up = NaN(1, length(HRchoicePerc));
    lo = NaN(1, length(HRchoicePerc));
    for ri = 1:length(HRchoicePerc)
        [up(ri), lo(ri)] = wilsonBinomialConfidenceInterval(HRchoicePerc(ri), z, nSamples(ri));
    end
    
else 
    up = [];
    lo = [];
    ee = [];
end


%% plot percentage HR vs stim rate.

if plotPMF
    
    ee = [(HRchoicePerc-lo)', (up-HRchoicePerc)'];
    ee = ee(~isnan(HRchoicePerc),:);

    xx = vec_rates(~isnan(HRchoicePerc)) + wd/2;
    yy = HRchoicePerc(~isnan(HRchoicePerc));
    
    
    %%
    if ~exist('lineColor', 'var')
        lineColor = 'k';
    end
    
%     figure;    
    hold on

    hmodal = plot(xx,yy,'.','color', lineColor);
    h = errorbar(xx, yy, ee(:,1), ee(:,2), 'color', lineColor,'linestyle','none');
    % errorbar_tick(h,200)

    % plot(vec_rates + wd/2, HRchoicePerc, 'k.')
    % h = errorbar(vec_rates + wd/2, HRchoicePerc, HRchoicePerc-lo, up-HRchoicePerc, 'color', 'k','linestyle','none');
    %{
    [h1,hp] = boundedline(x, y, e, 'alpha', 'transparency', .05);
    set(h1, 'color', 'k', 'marker', 'o');
    set(hp, 'facecolor', 'k');
    %}
    
    
    %% show the number of trials that went into each stim rate.
    
    if shownumtrs
        for ri = 1:length(vec_rates)
            text(vec_rates(ri)+wd/2, HRchoicePerc(ri)+.05, num2str(nSamples(ri)),'color', lineColor);
        end
    end
    
    
end

