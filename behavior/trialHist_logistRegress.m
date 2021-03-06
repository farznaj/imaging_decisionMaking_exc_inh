function [B_nan, deviance, stats_nan, X, y, B_inds, B_len, ...
    ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput, ...
    itiRatesDiffInput, itiSuccPrecedInput, itiFailPrecedInput, rateSuccPrecedInput, rateFailPrecedInput,...
    vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds, vec_ratesdiff2, ratesDiffVecInds2] = ...
    trialHist_logistRegress...
    (alldata, doiti, trials_per_session, binRates, binITIs, uncommittedResp, allowCorrectResp, ...
    doplots, vec_ratesdiff, vec_iti, excludeExtraStim, excludeShortWaitDur, regressModel, vec_ratesdiff2, mouse)
%
% [B, deviance, stats, X, y, vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds,...
%     ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput, vec_ratesdiff2, ratesDiffVecInds2] = trialHist_logistRegress...
%     (alldata, doiti, trials_per_session, binRates, binITIs, uncommittedResp, allowCorrectResp, ...
%     doplots, vec_iti, excludeExtraStim, excludeShortWaitDur, regressModel)
%
% trials_per_session: optional. If alldata is concatenated from several
% session, use this input to specify number of trials per session so you
% can identify and exclude the 1st trials of each session from the
% analysis.
%
% binRates: optional. If false actual stim rate values will be used in the
% regression model weiged by a constant regress coeff. If true(default),
% rate values will be binned and an indicator matrix will be used in the
% regression model.
%
% binITIs: optional. If false actual ITI values will be used in the
% regression model and a linear assumption will be made for how previous
% sucess(failure) varies with ITI. If true(default), ITI values will be
% binned and an indicator matrix will be used. No assumption on how
% previous outcome varies with ITI will be made.
%
% uncommittedResp: optional, could be: 'remove', 'change' or 'nothing'
% (default): how to deal with trials that mouse 1st made an uncommitted
% lick and then switched the response to the other side.
%
% allowCorrectResp: optional, could be 'remove', 'nothing', or 'change'
% (default): how to deal with trials that mouse entered allowCorrection
% state. Default changes the outcome and response side to the original lick
% (as if mouse was not allowed to correct).
%
% regressModel : 'rate_outcomeITI' (default), 'rate_outcomeRate',
% 'rateITI_outcomeITI', 'rateITI_outcomeRate'
% or set binITIs to 0 to do rate_outcomeITI on the actual ITI values.
% or set doiti to 0 to do rate_outcome model.

%%
if ~exist('doiti', 'var')
    doiti = false;
end

if ~exist('binRates', 'var')
    binRates = true;
end

if ~exist('binITIs', 'var')
    binITIs = true;
end

if exist('trials_per_session', 'var') && ~isempty(trials_per_session)
    % index of trial 1 in the concatenated alldata for each session
    begTrs = [0 cumsum(trials_per_session)]+1;
    begTrs = begTrs(1:end-1);
else
    begTrs = 1;
end

if ~exist('uncommittedResp', 'var')
    uncommittedResp = 'nothing'; % leave outcome and allResp as they are (ie go with the final choice of the mouse).
end

if ~exist('allowCorrectResp', 'var')
    allowCorrectResp = 'change'; % change the response on trials that entered allowCorrection to the original choice.
end

if ~exist('doplots', 'var')
    doplots = false;
end

if ~exist('excludeExtraStim', 'var')
    excludeExtraStim = false;
end

if ~exist('excludeShortWaitDur', 'var')
    excludeShortWaitDur = false;
end

if ~exist('regressModel', 'var') || isempty(regressModel)
    regressModel = 'rate_outcomeITI';
end

if ~exist('mouse', 'var') || isempty(mouse)
    mouse = '';
end


%% set trs2rmv (bad trials that you want to be removed from analysis) and stim rate

th = 5; % 10; 9; % this number of beginning trials will be excluded. Also later in the code we want >th trials in each column of ratesDiffInput
% trs2rmv = setTrs2rmv(alldata, th, excludeExtraStim, excludeShortWaitDur, begTrs);
[trs2rmv, stimdur, stimrate, stimtype, cb] = setTrs2rmv_final(alldata, th, excludeExtraStim, excludeShortWaitDur, begTrs);
% Use the code below if you are analyzing imaging data.
% imagingFlg = 1;
% [trs2rmv, stimdur, stimrate, stimtype, cb] = setTrs2rmv_final(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, imagingFlg, badAlignTrStartCode, trialStartMissing, trialCodeMissing);


% %% set stim rate
% [~, ~, stimrate, stimtype] = setStimRateType(alldata); % stimtype = [multisens, onlyvis, onlyaud];
% cb = unique([alldata.categoryBoundaryHz]); % categ boundary in hz
stimrate(trs2rmv) = NaN;


%% set outcome and response side for each trial, taking into account allcorrection and uncommitted responses.

% Remember for this analysis you use the outcome of the previous trial as a
% factor in decision-making, so it makes sense to use the final outcome of
% a trial (not the one before allowCorrection). This is why you set
% allowCorrectOutcomeChange to 0.
allowCorrectOutcomeChange = 0; % If 0, outcome of allowCorrEntered
% trials wont be changed (although animal's choice will be changed). If 1,
% outcome will be changed as well.
[outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp, allowCorrectOutcomeChange);
% Remember alldata.outcome is not necessarily the same as outcomes,
% depending on what you do to allCorrection trials, if it is set to change,
% then outcomes and alldata.outcome will be opposite to each other!!!


%%% set trs2rmv in the following vectors to NaN
outcomes(trs2rmv) = NaN;
% allResp(trs2rmv) = NaN;
allResp_HR_LR(trs2rmv) = NaN;


%% set the response (current trial's choice) vector: 0 for LR and 1 for HR.
% y(tr)=1 if tr was a HR choice, y(tr)=0 if tr was a LR choice, y(tr)=nan, if tr was a trs2rmv or had an outcome other than success, failure.

y = allResp_HR_LR';
% y(trs2rmv) = NaN;
% y(isnan(stimrate)) = NaN; % this is taken care of setTrs2rmv_final

% y(stimtype(:,1)~=1) = nan;

good_corr_incorr = ~isnan(y'); % trials that are not among trs2rmv and they ended up being either correct or incorrect. (doesn't include trials with any other outcome.)
% num_final_trs = sum(good_corr_incorr), fract_final_trs = mean(good_corr_incorr) % number and fraction of trials that will be used in the model fit.


%%
if sum(good_corr_incorr)<20
    warning('Too few trials remained after applying trs2rmv! aborting...')
    
    B_nan = []; deviance = []; stats_nan = []; X = []; y = []; B_inds = []; B_len = []; ...
    ratesDiffInput = []; successPrevInput = []; failurePrevInput = []; itiPrecedInput = []; ...
    itiRatesDiffInput = []; itiSuccPrecedInput = []; itiFailPrecedInput = []; rateSuccPrecedInput = []; rateFailPrecedInput = [];...
    vec_ratesdiff = []; ratesDiffVecInds = []; vec_iti = []; itiVecInds = []; vec_ratesdiff2 = []; ratesDiffVecInds2 = [];    
    return
end

if doplots
    figh = figure('name', ['Mouse ', mouse]);
else
    figh = [];
end


%%% set to nan those trials in stimrate and outcomes that are nan in y (choice vector).

% stimrate(isnan(y)) = nan;
% outcomes(isnan(y)) = nan;


%% set variables required for logistic regression.

[ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput, ...
    itiSuccPrecedInput, itiFailPrecedInput, vec_ratesdiff, ratesDiffVecInds, ...
    vec_iti, itiVecInds, num_fitted_trs, nan_y_s_f_t]...
  = trialHist_logisRegress_setVar ...
    (y, outcomes, stimrate, cb, alldata, doiti, binRates, binITIs, uncommittedResp,...
    doplots, vec_ratesdiff, vec_iti, begTrs, allowCorrectOutcomeChange, th, figh);


%% set bzero (constant) vector

bzeroInput = ones(length(alldata), 1);


%% set inputs for additional models.

if doiti && any(strcmp(regressModel, {'rate_outcomeRate', 'rateITI_outcomeITI', 'rateITI_outcomeRate'}))
    %% set new ratesDiffInput matrices for a new vector of rates.
    if ~exist('vec_ratesdiff2', 'var') || isempty(vec_ratesdiff2)
        vec_ratesdiff2 = [0 4 8 12];
    end
    
    [~, ~, bin_ratediff2] = histcounts(abs(stimrate-cb), vec_ratesdiff2);
    ratesDiffVecInds2 = unique(bin_ratediff2);
    ratesDiffVecInds2(~ratesDiffVecInds2) = []; % ind 0 corresponds to nans, so exclude it.
    
    
    %% set (stimrate - cb) input matrix (for current trial)
    
    ratesDiffInput2 = NaN(length(alldata), length(ratesDiffVecInds2));
    ratesDiffInput3 = NaN(length(alldata), length(ratesDiffVecInds2));
    
    for itr = find(~isnan(y))'    % set ratesDiffInput only for trials are not tr2rmv or that the outcome was 0 or 1.
        ratesDiffInput2(itr,:) = ismember(ratesDiffVecInds2, bin_ratediff2(itr)); % for each trial only one column will be non-zero, and that column index is the bin index of the trial's abs(stimrate-cb).
        ratesDiffInput3(itr,:) = ratesDiffInput2(itr,:); % no sign for HR and LR, just the strength
        ratesDiffInput2(itr,:) = sign(stimrate(itr)-cb) * ratesDiffInput2(itr,:); % assign 1 if the trial was HR and -1 if it was LR.
    end
    
    
    %%
    sr = stimrate(~isnan(y));
    n = histcounts(abs(sr-cb), vec_ratesdiff2);
    %     th = 9; % we want >th trials in each column of ratesDiffInput
    ratesDiffInput2 = ratesDiffInput2(:, ismember(ratesDiffVecInds2, find(n>th))); % only keep those columns of ratesDiffInput that have more than th trials.
    ratesDiffInput3 = ratesDiffInput3(:, ismember(ratesDiffVecInds2, find(n>th)));
    ratesDiffVecInds2 = find(n>th);
    
    % figure; plot(vec_ratesdiff2(ratesDiffVecInds2), n(ratesDiffVecInds2))
    
    
    %% previous outcome divided into columns based on current rateDiff
    
    rateSuccPrecedInput = bsxfun(@times, ratesDiffInput3, successPrevInput);
    rateFailPrecedInput = bsxfun(@times, ratesDiffInput3, failurePrevInput);
    % succ (and fail) vectors are divided into a few columns, each
    % corresponding to a stim strength. so the coef of each column is about the
    % effect of previous succ (or fail) for a particular stim strength.
    
    % 1 and -1 indicate previous succ_HR and succ_LR. column indicates current stim rate strength (ie abs(stimRate-cb) not its actual rate).
    
    
    %% current rate divided into columns based on ITI
    
    l = size(itiPrecedInput,2);
    itiRatesDiffInput = NaN(length(y), size(ratesDiffInput2, 2)*l);
    for it = 1:size(ratesDiffInput2, 2)
        e = it*l;
        r = e-l+1 : e;
        itiRatesDiffInput(:, r) = bsxfun(@times, itiPrecedInput, ratesDiffInput2(:,it));
    end
    % each rate column is divided into n columns, each for a bin of ITI.
    % so the coef of each column tell us about the effect of stim
    % rate given the ITI of the previous trial.
    
    % each column corresponds to an ITI bin. A 1 in column 2 indicates that
    % the trial was preceded by an ITI in bin2 of vec_iti also the trial
    % was a weak HR. A -1 indicates it was a weak LR.
    
    
    %% rateSuccPrecedInput, rateFailPrecedInput
    inds_rateSuccPrecedInput = ismember(1:length(vec_ratesdiff2)-1, ratesDiffVecInds2);
    len_rateSuccPrecedInput = length(vec_ratesdiff2)-1;
    
    % itiRatesDiffInput
    r = repmat(ismember(1:length(vec_ratesdiff2)-1, ratesDiffVecInds2), [length(vec_iti)-1,1]);
    r_t = bsxfun(@times, r, ismember(1:length(vec_iti)-1, itiVecInds)');
    inds_itiRatesDiffInput = r_t(:)';
    len_itiRatesDiffInput = (length(vec_ratesdiff2)-1) * (length(vec_iti)-1);
    
else
    itiRatesDiffInput = [];
    rateSuccPrecedInput = [];
    rateFailPrecedInput = [];
    ratesDiffVecInds2 = [];
end


%% prepare the indeces for different components of B_all
% ratesDiffInput
inds_ratesDiffInput = ismember(1:length(vec_ratesdiff)-1, ratesDiffVecInds);
len_ratesDiffInput = length(vec_ratesdiff)-1;

% itiSuccPrecedInput, itiFailPrecedInput
inds_itiSuccPrecedInput = ismember(1:length(vec_iti)-1, itiVecInds);
len_itiSuccPrecedInput = length(vec_iti)-1;



%% set the X (predictor) matrix for different models.

% ratesDiffInput : currect stimrate diff
% itiRatesDiffInput: each current stimrate diff bin divided into iti bins

% successPrevInput: previous successful choice
% failurePrevInput: previous failed choice

% itiSuccPrecedInput: previous success choice divided into iti bins
% itiFailPrecedInput: previous failed choice divided into iti bins

% rateSuccPrecedInput: previous success choice divided into current stim strength bins
% rateFailPrecedInput: previous failed choice divided into current stim strength bins

switch regressModel
    % look at the effect of 1) rate 2) previous outcome across itis
    case 'rate_outcomeITI'
        X = [bzeroInput, ratesDiffInput, itiSuccPrecedInput, itiFailPrecedInput];
        B_inds = [1, inds_ratesDiffInput, inds_itiSuccPrecedInput, inds_itiSuccPrecedInput];
        B_len = [1, len_ratesDiffInput, len_itiSuccPrecedInput, len_itiSuccPrecedInput];
        
    case 'rate_outcomeRate'
        % look at the effect of 1)rate 2) previous outcome across different current rates.
        X = [bzeroInput, ratesDiffInput, rateSuccPrecedInput, rateFailPrecedInput];
        B_inds = [1, inds_ratesDiffInput, inds_rateSuccPrecedInput, inds_rateSuccPrecedInput];
        B_len = [1, len_ratesDiffInput, len_rateSuccPrecedInput, len_rateSuccPrecedInput];
        
    case 'rateITI_outcomeITI'
        % look at the effect of 1) current rate across different itis 2) previous outcome across itis.
        X = [bzeroInput, itiRatesDiffInput, itiSuccPrecedInput, itiFailPrecedInput];
        B_inds = [1, inds_itiRatesDiffInput, inds_itiSuccPrecedInput, inds_itiSuccPrecedInput];
        B_len = [1, len_itiRatesDiffInput, len_itiSuccPrecedInput, len_itiSuccPrecedInput];
        
    case 'rateITI_outcomeRate'
        % look at the effect of 1) current rate across different itis  2) previous outcome across current rates.
        X = [bzeroInput, itiRatesDiffInput, rateSuccPrecedInput, rateFailPrecedInput];
        B_inds = [1, inds_itiRatesDiffInput, inds_rateSuccPrecedInput, inds_rateSuccPrecedInput];
        B_len = [1, len_itiRatesDiffInput, len_rateSuccPrecedInput, len_rateSuccPrecedInput];
        
    otherwise
        if doiti
            if ~binITIs % actual ITI values used.
                X = [bzeroInput, ratesDiffInput, successPrevInput, failurePrevInput, itiSuccPrecedInput, itiFailPrecedInput]; % I think this makes the most sense in terms of iti.
                B_inds = [1, inds_ratesDiffInput, 1, 1, inds_itiSuccPrecedInput, inds_itiSuccPrecedInput];
                B_len = [1, len_ratesDiffInput, 1, 1, len_itiSuccPrecedInput, len_itiSuccPrecedInput];
                
                %     else % ITI values binned and an indicator matrix used.
                % you commented this bc it is the 'rate_outcomeITI' model.
                %         X = [bzeroInput, ratesDiffInput, itiSuccPrecedInput, itiFailPrecedInput]; % I think this makes the most sense in terms of iti.
            end
        else % rate_outcome model
            X = [bzeroInput, ratesDiffInput, successPrevInput, failurePrevInput]; % similar to Busse paper.
            B_inds = [1, inds_ratesDiffInput, 1, 1];
            B_len = [1, len_ratesDiffInput, 1, 1];
        end
end
B_inds = logical(B_inds);

% sum(B_inds) == size(X,2)
% figure; imagesc(X)
% [sum(X == 1)', sum(X == -1)']


%% fit the data with a logistic regression, binomial model

if num_fitted_trs > 20
    
    lastwarn('')
    
    [B, deviance, stats] = glmfit(X, y, 'binomial', 'constant', 'off', 'estdisp', 'on');
    
    % exclude data if iteration limit reached in glmfit.
    if strcmp(lastwarn, 'Iteration limit reached.') || ...
            ~isempty(regexp(lastwarn, 'The estimated coefficients perfectly separate', 'once')) || ...
            ~isempty(regexp(lastwarn, 'X is ill conditioned', 'once'))
        warning('glmfit did not work. Setting B and stats to []!')
        disp(lastwarn)
        
        B = []; deviance = []; stats = [];
        B_nan = []; stats_nan = []; 
    else
        % use nan for those columns of rate or iti that had no trial in them. so its length is same as what u'd have expected if there were enough trials in all bins.
        B_nan = NaN(1, length(B_inds));
        B_nan(B_inds) = B;
        
        stats_nan = stats;
        
        stats_nan.beta(B_inds) = stats.beta;
        stats_nan.beta(~B_inds) = NaN;
        
        stats_nan.covb = ones(length(B_inds), length(B_inds));
        stats_nan.covb(~B_inds, :) = NaN;
        stats_nan.covb(:, ~B_inds) = NaN;
        stats_nan.covb(stats_nan.covb==1) = stats.covb;
        
        stats_nan.se(B_inds) = stats.se;
        stats_nan.se(~B_inds) = NaN;
        
        stats_nan.coeffcorr = ones(length(B_inds), length(B_inds));
        stats_nan.coeffcorr(~B_inds, :) = NaN;
        stats_nan.coeffcorr(:, ~B_inds) = NaN;
        stats_nan.coeffcorr(stats_nan.coeffcorr==1) = stats.coeffcorr;
        
        stats_nan.t(B_inds) = stats.t;
        stats_nan.t(~B_inds) = NaN;
        
        stats_nan.p(B_inds) = stats.p;
        stats_nan.p(~B_inds) = NaN;
        
    end
    
else
    warning('Too few trials to model! aborting...')
    B = []; deviance = []; stats = [];
end




%%
if doplots
    
    if isempty(B) % fitting did not work (iteration limit reached, etc, usually a result of not having enough number of trials.)
        figure(figh)
        subplot(428)
        PMF_set_plot(stimrate, y, cb, [], 1, 1);
        
    else
        
        % plot all coefficients
        figure(figh)
        subplot(427)
        errorbar(1:length(B), B, stats.se, 'k.')
        ylabel('Regress Coef')
        xlim([0 length(B)+1])
        
        
        %% evaluate the model
        figure(figh); subplot(424)
        normplot(stats.residp)
        
        fprintf('deviance = %.2f\n', deviance)
        
        pvals = stats.p;
        fprintf(['p values: ', repmat('%.2f  ', 1, length(pvals)), '\n'], pvals)
        
        % if you want to know how some of the fields of stats are computed.
        % isequal(sum(~isnan(y)) - size(X,2), stats.dfe)
        % isequal(sqrt(diag(stats.covb)), stats.se)
        % isequal(B ./ stats.se, stats.t)
        %{
        if estdisp
            stats.p = 2 * tcdf(-abs(stats.t), dfe);
        else
            stats.p = 2 * normcdf(-abs(stats.t));
        end
        %}
        
        
        %% plot the coefficients of the model
        
        % overall bias coef
        fprintf('overall Bias Coef = %.2f\n', B(1))
        
        css = [0 cumsum(B_len)];
        
        B_term = cell(1, length(B_len));
        for ib = 1:length(B_len)
            B_term{ib} = B_nan(css(ib)+1:css(ib+1));
        end
        
        se_term = cell(1, length(B_len));
        for ib = 1:length(B_len)
            se_term{ib} = stats_nan.se(css(ib)+1:css(ib+1));
        end
        
        figure(figh)
        % plot rate and iti coeffs.
        col = {'k','r','g', 'b', 'm', 'c'};
        for ib = 2%:length(B_term)
            subplot(4,2,5)
            errorbar(1:size(B_term{ib},2), B_term{ib}, se_term{ib}, 'k.');
            ylabel('Regress coeff')
        end
        
        clear l
        cnt = 0;
        for ib = 3:length(B_term)
            cnt = cnt+1;
            subplot(4,2,6), hold on
            l(cnt) = errorbar(1:size(B_term{ib},2), B_term{ib}, se_term{ib}, '.', 'color', col{ib});
            ylabel('Regress coeff')
        end
        legend(l)
        
        %{
    % stimRate coef
    if binRates  % plot coeffs of different stim rates.
        b = 2;
        e = b + length(ratesDiffVecInds)-1;
        x = vec_ratesdiff(ratesDiffVecInds)+wd/2;
        
        subplot(425)
        errorbar(x, B(b:e), stats.se(b:e), 'k.-')
        %     plot(x, B(b:e), 'o-')
        xlabel('Rate difference from categ. bound. (Hz)')
        ylabel('Regress Coeff')
        xlim([x(1)-1 x(end)+1])
        
    else
        stimRateCoef = B(2)
    end
    
    
    if doiti
        % ITI coef
        b = 1 + length(ratesDiffVecInds) +1;
        if ~binITIs
            subplot(212); hold on
            plot(sort(itiPreced), B(b)+B(b+2)*sort(itiPrecedN), 'k.')
            plot(sort(itiPreced), B(b+1)+B(b+3)*sort(itiPrecedN), 'r.')
            plot([min(itiPreced) max(itiPreced)], [0 0], ':k')
            legend('b\_succ','b\_fail')
            
        else
            % plot coeffs of itiSucc and itiFail.
            e = b + length(itiVecInds)-1;
            
            subplot(426); hold on;
            %         boundedline(vec_iti(itiVecInds), B(b:e), stats.se(b:e), 'k.-', 'alpha')
            %         boundedline(vec_iti(itiVecInds), B(e+1:end), stats.se(b:e), 'r.-', 'alpha')
            errorbar(vec_iti(itiVecInds), B(b:e), stats.se(b:e), 'k.-')
            errorbar(vec_iti(itiVecInds), B(e+1:end), stats.se(b:e), 'r.-')
            
            plot([vec_iti(itiVecInds(1)) vec_iti(itiVecInds(end))], [0 0], ':k')
            
            legend('b\_succ','b\_fail'), set(legend, 'location', 'southeast', 'box', 'off')
            xlabel('ITI(sec)')
            ylabel('Regress Coeff')
            set(gca, 'xtick', vec_iti(itiVecInds))
            
            l = round(vec_iti(itiVecInds));
            lab = cell(1, length(l));
            for i=1:length(l)-1
                lab{i} = [sprintf('%d-%d', l(i), l(i+1))];
            end
            lab{i+1} = [sprintf('>=%d', l(end))];
            set(gca, 'xticklabel', lab)
        end
    end
        %}
        
        
        %% make PMF based on the model predictions
        
        % xs = (rates(1) : 0.1 : rates(end))';
        % yfit = glmval(B, xs, 'logit');
        
        z = X*B;
        p = 1 ./ (1+exp(-z));
        
        yfit = p; % animal's choice driven from the regression model.
        yfit(p>.5) = 1; % 1: HR
        yfit(p<.5) = 0; % 0: LR
        
        compModel = yfit - y;
        c = compModel(~isnan(compModel));
        fprintf('Model prediction: incorrect = %.2f, correct = %.2f\n', ...
            [mean(abs(c)==1), mean(c==0)])
        
        
        y(nan_y_s_f_t) = NaN;
        plotPMF = true;
        shownumtrs = false; %true;
        
        figure(figh)
        subplot(428)
        PMF_set_plot(stimrate, y, cb, [], plotPMF, shownumtrs);
        PMF_set_plot(stimrate, p, cb, [], plotPMF, 0, 'r');
        PMF_set_plot(stimrate, yfit, cb, [], plotPMF, 0, 'y');
        xlabel('Stim rate (Hz)')
        ylabel('Prop HR choice')
        xlim([min(stimrate)-1 max(stimrate)+1])
        ylim([-.1 1.1])
        f1 = findobj(gca, 'color', 'k', 'type', 'line');
        f2 = findobj(gca, 'color', 'r', 'type', 'line');
        f3 = findobj(gca, 'color', 'y', 'type', 'line');
        legend([f1(1), f2(1), f3(1)], 'data','fitted P','fitted category')
        
    end
    
end

%{
% the result of the code below is similar to : PMF_set_plot(stimrate, p, cb, [], plotPMF, 0, 'r');
% flip a coin, use p for each trial to decide if the coin is HR or LR.
% repeat this 1000 iterations, and take average of propHRs to plot PMF.

y_sim = NaN(length(y), 1000);
for iter = 1:1000
    r = rand(length(y), 1);
    y_bernoulli = double(r<=p);
    y_bernoulli(nan_y_s_f_t) = NaN;
    y_sim(:, iter) = y_bernoulli;
end

yy_sim = NaN(size(y_sim,2), sum(~isnan(HRchoicePerc)));
for iter = 1:1000
    [HRchoicePerc, vec_rates, up, lo, nSamples, xx, yy, ee] = PMF_set_plot(stimrate, y_sim(:,iter), cb, [], 0, 0, 0);
    yy_sim(iter,:) = yy;
end
yya = nanmean(yy_sim);

z = 1; % 1 SEM
up = NaN(1, length(yya));
lo = NaN(1, length(yya));
for ri = 1:length(yya)
    [up(ri), lo(ri)] = wilsonBinomialConfidenceInterval(yya(ri), z, nSamples(ri));
end
ee = [(yya-lo)', (up-yya)'];

figure; hold on
plot(xx, yya, '.','color', 'k');
h = errorbar(xx, yya, ee(:,1), ee(:,2), 'color', 'k','linestyle','none');
%}


%% old method for setting s_prev and f_prev, correct but u wrote a much better one above.
%{
if strcmp(alldata(1).highRateChoicePort, 'L')
    allRespHR = (allResp==1); % left
    allRespLR = (allResp==2); % right
elseif strcmp(alldata(1).highRateChoicePort, 'R')
    allRespHR = (allResp==2); % right
    allRespLR = (allResp==1); % left
end

%% set success vector for the previous trial

outs = outcomes; % [alldata.outcome];
% outs(trs2rmv) = NaN;
fract_corrBut1stErr_ErrBut1stCorr = [sum(outs==1 & errorlick_again_wait_entered) / sum(outs==1) ...
    sum(outs==0 & correctlick_again_wait_entered) / sum(outs==0)]

% you don't want to call a trial correct if it was preceded by
% an error lick (even if uncommitted).
% outs(errorlick_again_wait_entered) = NaN;
correct = (outs==1);

sLR = correct & allRespLR;
sHR = correct & allRespHR;
numCorr_LR_HR = [sum(sLR), sum(sHR)]


%%%% set success vector for the previous trial: if 1, previous trial was a
%%%% sucess HR, if -1 previous trial was a success LR.
successPrevInput = zeros(length(alldata), 1);

% previous trial: LR side --> -1
f = find(sLR)+1; % index of trials following a success LR choice.
f(f>length(alldata)) = [];
successPrevInput(f) = -1;

% previous trial: HR side --> 1
f = find(sHR)+1; % index of trials following a success HR choice.
f(f>length(alldata)) = [];
successPrevInput(f) = 1;

successPrevInput(find(~good_corr_incorr)+1) = NaN;  % set to NaN trial following a tr2rmv or a trial with an outcome other than 0 or 1. % you don't need to do this because Y has NaN, and glmfit will ignore these rows anyways.
successPrevInput(begTrs) = NaN; % no previous trial exists for the 1st trial of a session.
successPrevInput(length(y)+1:end) = [];
% figure; imagesc(successPrevInput)


%% set failure vector for the previous trial
outs = outcomes; % [alldata.outcome];
% outs(trs2rmv) = NaN;

% you don't want to call a trial incorrect if it was preceded by
% a correct lick (even if uncommitted).
% outs(correctlick_again_wait_entered) = NaN;
incorrect = (outs==0);

fLR = incorrect & allRespLR;
fHR = incorrect & allRespHR;
numIncorr_LR_HR = [sum(fLR), sum(fHR)]


%%%%
failurePrevInput = zeros(length(alldata), 1);

% previsou trials: LR side --> -1
f = find(fLR)+1; % index of trials following a failure LR choice.
f(f>length(alldata)) = [];
failurePrevInput(f) = -1;

% previous trial: HR side --> 1
f = find(fHR)+1; % index of trials following a failure HR choice.
f(f>length(alldata)) = [];
failurePrevInput(f) = 1;

failurePrevInput(find(~good_corr_incorr)+1) = NaN;  % set to NaN trial following a trial with an outcome other than 0 or 1. or following a tr2rmv trial. (bc these trs are all NaN in Y, hence in corr_incorr)
failurePrevInput(begTrs) = NaN; % no previous trial exists for the 1st trial of a session.
failurePrevInput(length(y)+1:end) = [];
% figure; imagesc(failurePrevInput)


%}