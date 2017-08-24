function [ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput, ...
    itiSuccPrecedInput, itiFailPrecedInput, vec_ratesdiff, ratesDiffVecInds, ...
    vec_iti, itiVecInds, num_fitted_trs, nan_y_s_f_t]...
    = trialHist_logisRegress_setVar ...
    (y, outcomes, stimrate, cb, alldata, doiti, binRates, binITIs, uncommittedResp,...
    doplots, vec_ratesdiff, vec_iti, begTrs, allowCorrectOutcomeChange, th, figh)
% Sets variables that are used in logisitc regression.


%%
if ~exist('figh', 'var')
    figh = figure;
end


good_corr_incorr = ~isnan(y'); % trials that are not among trs2rmv and they ended up being either correct or incorrect. (doesn't include trials with any other outcome.)


%% set vars for the stim rate matrix.

if ~binRates
    ratesDiffInput = stimrate-cb;
    ratesDiffInput = ratesDiffInput / max(abs(ratesDiffInput)); % normalize it so it is in the same range as other input vectors (success, failure, etc).
    
else
    if ~exist('vec_ratesdiff', 'var') || isempty(vec_ratesdiff)
        wd = 2;
        vec_ratesdiff = 0: wd: max(stimrate)+wd-cb; % 0: wd: max(abs(stimrate-cb))+wd
        % vec_rates = sort([cb : -wd : min(stimrate)-wd  ,  cb+wd : wd : max(stimrate)+wd]); % make sure categ boundary doesn't go at the middle of a bin!
        % [n, bin_rate] = histc(stimrate, vec_rates);
    else
        wd = unique(diff(vec_ratesdiff));
    end
    
    [~,~, bin_ratediff] = histcounts(abs(stimrate-cb), vec_ratesdiff);
    ratesDiffVecInds = unique(bin_ratediff);
    ratesDiffVecInds(~ratesDiffVecInds) = []; % ind 0 corresponds to nans, so exclude it.
    
    
    %% set (stimrate - cb) input matrix (for current trial)
    
    % ratesDiffInput = zeros(length(alldata), length(ratesDiffVecInds));
    ratesDiffInput = NaN(length(alldata), length(ratesDiffVecInds));
    
    % for itr = 1:length(alldata)
    for itr = find(good_corr_incorr)    % set ratesDiffInput only for trials are not tr2rmv or that the outcome was 0 or 1.
        ratesDiffInput(itr,:) = ismember(ratesDiffVecInds, bin_ratediff(itr)); % for each trial only one column will be non-zero, and that column index is the bin index of the trial's abs(stimrate-cb).
        ratesDiffInput(itr,:) = sign(stimrate(itr)-cb) * ratesDiffInput(itr,:); % assign 1 if the trial was HR and -1 if it was LR.
    end
    % If ratesDiffInput(tr,rv)=1 or -1, then trial tr's abs(stimrate-cb) was in
    % bin rv of vec_ratesdiff, ie within [vec_ratesdiff(rv) vec_ratesdiff(rv+1)-eps]
    %
    % If ratesDiffInput(tr,rv)=0, then trial tr's abs(stimrate-cb) was NOT in
    % bin rv of vec_ratesdiff, EXCEPT for when stimrate(tr)=cb. In this case
    % ratesDiffInput(tr,rv) should be 1 or -1 but bc you multiply it by
    % sign(stimrate(itr)-cb), it gets 0. So 0s in ratesDiffInput(:,1) are
    % either bc the trial's abs(stimrate-cb) was in a bin other than rv, or bc
    % its abs(stimrate-cb) was 0.
    
    % figure; imagesc(ratesDiffInput)
    
end


%% set success vector for the previous trial

% Note regarding how prev sucess and failure vectors are set: Remember for
% the explanation below, you need to look at outcomes and not
% alldata.outcome (they could be different if you set allowCorrectResp to
% change and allowCorrectOutcomeChange to 1). See above for more detialed
% explanation. successPrevInput(i) indicates the success outcome and choice
% of trial i-1: if successPrevInput(i)=1, then trial i-1 was a successful
% HR-choice trial. if successPrevInput(i)=-1, then trial i-1 was a
% successful LR-choice trial. if successPrevInput(i)=0, then trial i-1 was
% a failure trial.
%
% failurePrevInput(i) indicates the failure outcome and choice of trial i-1:
% if failurePrevInput(i)=1, then trial i-1 was a failure HR-choice
% trial.
% if failurePrevInput(i)=-1, then trial i-1 was a failure LR-choice
% trial.
% if failurePrevInput(i)=0, then trial i-1 was a successful trial.
%


numPrevTrs = 1; % 1 : what trial in the back you will study. If 3, it means you will study the effect of the 3rd trial in the past on current choice.

% set s_curr indicating the success/failure and HR/LR outcome of the
% current trial.
ynew = y;
ynew(y==0) = -1; % in ynew LR will be -1 and HR will remain 1.
s_curr = ynew .* (outcomes==1)'; % 0 or nan for any output other than success. 1 for succ HR. -1 for succ LR.

% set s_prev indicating the success/failure and HR/LR outcome of the
% previous trial. shift s_curr one element front.
successPrevInput = [NaN(numPrevTrs,1); s_curr(1:end-numPrevTrs)]; % s_prev(i) = s_curr(i-1);
successPrevInput(begTrs) = NaN; % no previous trial exists for the 1st trial of a session.

a = find(~good_corr_incorr); % these trials will be nan in y and ratesDiffInput
b = find(isnan(successPrevInput)); % these trials will be nan in successPrevInput and failurePrevInput
nan_y_s_f = union(a,b); % trials that are nan either in y or succ/fail vectors.
% num_fitted_trs = length(y) - length(nan_y_s_f) % final number of trials that will be using in the fitting.


%% set failure vector for the previous trial

% set f_curr indicating the success/failure and HR/LR outcome of the
% current trial.
f_curr = ynew .* (outcomes==0)'; % 0 or nan for any output other than failure. 1 for fail HR. -1 for fail LR.

% set f_prev indicating the success/failure and HR/LR outcome of the
% previous trial. shift f_curr one element front.
failurePrevInput = [NaN(numPrevTrs,1); f_curr(1:end-numPrevTrs)]; % s_prev(i) = s_curr(i-1);
failurePrevInput(begTrs) = NaN; % no previous trial exists for the 1st trial of a session.


%% take care of ITI input

if doiti
    %% compute ITI
    % many ways to define it. for now: iti = time between commiting a choice in
    % trial i and stimulus onset in trial i+1.
    % if no committed choice or no stim onset set iti to nan.
    
    % how about iti = stim onset to stim onset? you can plot the two.
    
    % time of making choice
    t0 = NaN(length(alldata), 1);
    if strcmp(uncommittedResp, 'change')
        % use time of 1st lick (not necessarily got committed but you defined outcome based on it.).
        for itr = 1:length(alldata)
            if outcomes(itr)==1
                t0(itr) = alldata(itr).parsedEvents.states.correctlick_again_wait(1);
                
            elseif outcomes(itr)==0
                t0(itr) = alldata(itr).parsedEvents.states.errorlick_again_wait(1);
            end
        end
        
    else % use time of commit response
        if allowCorrectOutcomeChange % change outcome of allowCorrect trials.
            error('Not sure why you are not changing the outcome of allowCorrect trials! BUT you need to define t0 based on the last committed choice (after allowCorrection). Remember for this analysis you use outcome of previous trial as a factor in decision-making, so it makes sense to use the final outcome of a trial (not the one before allowCorrection.)')
        end
        for itr = 1:length(alldata)
            % if allowCorrectOutcomeChange is set to 0, in
            % allowCorrectEntered trials outcome is defined based on the
            % final outcome (not the outcome of the 1st choice). As a
            % result t0 will be defined based on the time of last committed
            % choice (not the 1st committed choice).
            if outcomes(itr)==1
                t0(itr) = alldata(itr).parsedEvents.states.reward(1);
                
            elseif outcomes(itr)==0
                if ~isempty(alldata(itr).parsedEvents.states.punish)
                    t0(itr) = alldata(itr).parsedEvents.states.punish(1);
                    
                else % punish allow correction entered.
                    t0(itr) = alldata(itr).parsedEvents.states.punish_allowcorrection(end,1);
                    %                     pac = alldata(itr).parsedEvents.states.punish_allowcorrection;
                    %                     pacd = alldata(itr).parsedEvents.states.punish_allowcorrection_done;
                    %                     t0(itr) = pac(ismember(pac(:,2), pacd(1)),1);       % this is not needed. mouse commits his wrong choice once he enters the pucnish_allcorrection state.
                end
            end
        end
    end
    
    % time of stim onset
    t1 = NaN(length(alldata), 1);
    for itr = 1:length(alldata)
        if ~isempty(alldata(itr).parsedEvents.states.wait_stim) % ~ismember(outcomes(itr), -3) % wrong init.
            t1(itr) = alldata(itr).parsedEvents.states.wait_stim(1);
        end
    end
    
    % itiProced will be NaN if a trial was wrong init or if its preceding trial
    % did not have a committed choice.
    itiPreced = NaN(length(alldata), 1);
    itiPreced(2:end) = t1(2:end) - t0(1:end-1); % in sec, stimOn of current trial - choiceCommit of previous trial
    itiPreced(begTrs) = NaN; % no previous trial exists for the 1st trial of a session.
    %     min_max_iti = [min(itiPreced)  max(itiPreced)]
    
    % On Nov3,2016 I commented the line below (itiPreced(nan_y_s_f) = NaN),
    % because for SVM analysis you use itiSuccPrecedInput (defined using
    % itiPreced), and in itiSuccPrecedInput you don't want to set to nan
    % trials whose current outcome is other than correct or incorrect,
    % since you don't care about their current outcome. On the other hand
    % for the logistic regression analysis of behavior, as explained below,
    % these trials wont be anyhow fitted because their y entry is nan. So
    % no need to do itiPreced(nan_y_s_f) = NaN!
%     itiPreced(nan_y_s_f) = NaN; % you don't need to do this bc these trials wont be anyway fitted (they're nan in y or s or f), but u do it to make the iti bins fewer.
    fprintf('ITI, min: %.2f, max: %.2f\n', min(itiPreced), max(itiPreced))
    
    if doplots
        figure(figh); subplot(421),
        histogram(itiPreced)
        xlabel('ITI (sec)')
        ylabel('number of trials')
    end
    
    %     warning('define the outlier value for iti... using arbitrary values...')
    %     itiPreced(itiPreced > 50) = NaN; % 3*quantile(itiPreced, .9)
    fprintf('%d trials have ITI > 30ms, excluding them .... \n', sum(itiPreced > 30))
    successPrevInput(itiPreced > 30) = NaN; % added on Nov3,2016
    failurePrevInput(itiPreced > 30) = NaN; % added on Nov3,2016    
    itiPreced(itiPreced > 30) = NaN; % 3*quantile(itiPreced, .75)
    %     itiPreced(itiPreced > 15) = NaN; % 3*nanstd(itiPreced)
    
    c = find(isnan(itiPreced));
    nan_y_s_f_t = union(nan_y_s_f, c); % trials that are nan either in y or succ/fail vectors.
    
    
    %% set ITI matrix
    
    if ~binITIs % go with the actual values of ITI.
        itiPrecedN = itiPreced / max(itiPreced); % normalize it so it is in the same range as other input vectors (success, failure, etc).
        itiSuccPrecedInput = itiPrecedN .* successPrevInput;
        itiFailPrecedInput = itiPrecedN .* failurePrevInput;
        
    else % bin the ITIs
        if ~exist('vec_iti', 'var') || isempty(vec_iti)
            %         vec_iti = floor(min(itiPreced)) : ceil(max(itiPreced));
            %         vec_iti = [min(itiPreced)-1 6 9 12 max(itiPreced)+1];
            %         vec_iti = [min(itiPreced)-1 6 9 max(itiPreced)+1];
            %         vec_iti = [min(itiPreced)-1 7 9 max(itiPreced)+1];
            vec_iti = [min(itiPreced)-1 7 max(itiPreced)+1];
            %         vec_iti = [min(itiPreced)-1 10 max(itiPreced)+1];
            %         vec_iti = [min(itiPreced)-1 max(itiPreced)+1];
        end
        
        
        %%
        [~, bin_iti] = histc(itiPreced, vec_iti);
        itiVecInds = unique(bin_iti);
        itiVecInds(~itiVecInds) = []; % bin_iti=0 corresponds to NaN values of itiPreced. exclude them.
        
        % I think it makes more sense to define it as below, so
        % itiPrecedInput has always a consistent size even if a particular
        % day doesnt have any trials in a bin of vec_iti
        
%         itiVecInds = 1:length(vec_iti)-1;
        
        itiPrecedInput = NaN(length(alldata), length(itiVecInds));
        for itr = 1:length(alldata)
            itiPrecedInput(itr,:) = ismember(itiVecInds, bin_iti(itr)); % for each trial only one column will be non-zero, and that column index is the bin index of the trial's iti.
        end
        itiPrecedInput(isnan(itiPreced),:) = NaN; % try not doing this and setting all columns to 0 for these trials.
        
        %         itiHRLRPrecedInput = bsxfun(@times, itiPrecedInput, successPrevInput+failurePrevInput);
        % figure; imagesc(itiHRLRPrecedInput)
        % (nansum(abs(itiHRLRPrecedInput)))
        
        itiSuccPrecedInput = bsxfun(@times, itiPrecedInput, successPrevInput);
        % figure; imagesc(itiSuccPrecedInput)
        
        itiFailPrecedInput = bsxfun(@times, itiPrecedInput, failurePrevInput);
        % figure; imagesc(itiFailPrecedInput)
        
        
        %% number of trials that went into each bin for iti_succ and iti_fail
        
        if doplots
            figure(figh); subplot(422), hold on
            plot(vec_iti(itiVecInds), nansum(abs(itiSuccPrecedInput)), 'o-')
            plot(vec_iti(itiVecInds), nansum(abs(itiFailPrecedInput)), 'o-')
            xlabel('ITI (sec)')
            ylabel('number of trials')
            legend('itiS', 'itiF')
        end
        
    end
else
    nan_y_s_f_t = nan_y_s_f;
end


%% set number of trials that will go into the model.

num_fitted_trs = length(y) - length(nan_y_s_f_t); % final number of trials that will be using in the fitting.
fprintf('Number of trials, original: %d, fitted: %d\n', length(y), num_fitted_trs)


%% now that all nan trials are determined, make sure that there is no column of ratesDiffInput with <=th trials.

if binRates
    sr = stimrate(~isnan(y));
    n = histcounts(abs(sr-cb), vec_ratesdiff);
    %     th = 9; % we want >th trials in each column of ratesDiffInput
    ratesDiffInput = ratesDiffInput(:, ismember(ratesDiffVecInds, find(n>th))); % only keep those columns of ratesDiffInput that have more than th trials.
    ratesDiffVecInds = find(n>th);
else
    vec_ratesdiff = [];
    ratesDiffVecInds = [];
end

if ~binITIs
    vec_iti = [];
    itiVecInds = [];
end


%% plot number of trials in each bin of vec_ratesdiff that will be fitted.

if doplots
    figure(figh); subplot(423)
    plot(vec_ratesdiff(ratesDiffVecInds)+wd/2, n(n>th), '.-')
    xlabel('abs(stimRate - cb)')
    ylabel('number of trials')
    xlim([vec_ratesdiff(1)  vec_ratesdiff(end)])
end

