% This is the main starting script for the logistic regression analysis of
% behavioral data (predicting animals choice from a number of factors like
% stimulus rate, ITI and previous choice).
% It sets parameters and calls trialHist_logistRegress.
% It pools trials from all sessions for each mouse.


%%
miceNames = {'fn03', 'fn04', 'fn05', 'fn06', 'fni16', 'fni17', 'fni18', 'fni19', 'hni01', 'hni04'};

% For allowCorrectResp: change seems to be the best. nothing is fine, except B_rate for easy
% stimuli looks bad which makes sense. remove is fine too, you just get
% fewer mice.
allowCorrectResp = 'change'; % 'nothing'; %'remove'; % 'change'; % if you set to change, only allResp will change, but outcome will remain the same (this is important because if a trial was at the end successful, we think it should cause bias on the next trial, even if it was an allow correction trial).
binningRates = 0; 1; % this is for the conventional analysis, if true the effect of outcome will be shown for different stim strength.

doplots = false; % true; % make plots for each mouse for the logistic regression analysis.
doplots_conventAnalysis = false; % true; % make plots for each mouse for the conventional analysis.

regressModel = 'rate_outcomeITI'; % 'rate_outcomeITI' (default), 'rate_outcomeRate', 'rateITI_outcomeITI', 'rateITI_outcomeRate'
doiti = true;
binRates = true;
binITIs = true; % false; %true;

uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';

excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;

vec_iti = [0 10 30]; %[0 6 9 12 30]; % [0 7 30]; % [0 10 30]; % [0 6 9 12 30]; % use [0 40]; if you want to have a single iti bin and in conventioinal analysis look at the effect of current rate on outcome.
vec_ratesdiff = 0:2:12;
vec_ratesdiff2 = [0 6 10 12];

defaultHelpedTrs = false; % false; % set to 0 if you want to manually set the helped trials.
saveHelpedTrs = false; % true


%%
dev_all = NaN(1, length(miceNames));
% clear stats_all B_all
% for model 'rate_outcomeITI', number of coeffs will be 11.
B_all = NaN(length(miceNames), 20); % each row includes all coefficients of logisitic regression for each mouse. since the number of coeffs depends on the model, we go with 20 (a large number), but then later we remove columns that are nan for all mice.
clear stats_all
stats_all(length(miceNames)) = struct();
[stats_all.beta] = deal(NaN(20,1)); % again we go with a big estimate for the number of coefficients.
[stats_all.dfe] = deal(NaN);
[stats_all.sfit] = deal(NaN);
[stats_all.s] = deal(NaN);
[stats_all.estdisp] = deal(NaN);
[stats_all.covb] = deal(NaN(20,20));
[stats_all.se] = deal(NaN(20,1));
[stats_all.coeffcorr] = deal(NaN(20,20));
[stats_all.t] = deal(NaN(20,1));
[stats_all.p] = deal(NaN(20,1));
[stats_all.resid] = deal(NaN); % the accurate code is NaN(trials_per_mouse(icount),1)
[stats_all.residp] = deal(NaN); % the accurate code is NaN(trials_per_mouse(icount),1)
[stats_all.residd] = deal(NaN); % the accurate code is NaN(trials_per_mouse(icount),1)
[stats_all.resida] = deal(NaN); % the accurate code is NaN(trials_per_mouse(icount),1)
[stats_all.wts] = deal(NaN); % it should be length(y), ie number of fitted trials, but don't know why sometimes there is a small discrepency.

if strcmp(regressModel, 'rate_outcomeITI')
    fract_change_choosingSameChoice_aftS_all = NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingSameChoice_aftF_all = NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingHR_aftHR_vs_LR_S_all = NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingLR_aftLR_vs_HR_S_all = NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingHR_aftHR_vs_LR_F_all = NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingLR_aftLR_vs_HR_F_all = NaN(length(miceNames), size(vec_iti,2)-1);
else
    fract_change_choosingSameChoice_aftS_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingSameChoice_aftF_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingHR_aftHR_vs_LR_S_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingLR_aftLR_vs_HR_S_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingHR_aftHR_vs_LR_F_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
    fract_change_choosingLR_aftLR_vs_HR_F_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
end

sess_num = 0;
trials_per_mouse = NaN(1, length(miceNames));


%%
for icount = 1:length(miceNames)
    
    fprintf('----------------------------------------------------\n')
    mouse = miceNames{icount};
    % if you want to analyze only 1 session of a mouse:
    %{
%     day = {datestr(datenum(imagingFolder, 'yymmdd'))};
    [alldata_fileNames, days_all] = setBehavFileNames(mouse, day);
    % sort it
    [~,fn] = fileparts(alldata_fileNames{1});
    a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
    [~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
    alldata_fileNames = alldata_fileNames(isf);
    % load the one corresponding to mdffilenumber.
    [alldata, trials_per_session] = loadBehavData(alldata_fileNames(mdfFileNumber)); % , defaultHelpedTrs, saveHelpedTrs); % it removes the last trial too.
    %}    
    
    % if you want to analyze all sessions of a mouse specifified in setMouseAnalysisDays
    [day, dayLast, days2exclude] = setMouseAnalysisDays(mouse);    
    [alldata_fileNames, days_all] = setBehavFileNames(mouse, day, dayLast, days2exclude);
    fprintf('Total number of sessions: %d\n', length(alldata_fileNames))
    
    % load alldata
    [alldata, trials_per_session] = loadBehavData(alldata_fileNames, defaultHelpedTrs, saveHelpedTrs);
    fprintf('Total number of trials: %d\n', length(alldata))
    trials_per_mouse(icount) = length(alldata);
    

    %% Do logisitic regression
    
    clear B deviance stats X y B_inds B_len ratesDiffInput successPrevInput failurePrevInput itiPrecedInput
    
    [B, deviance, stats, X, y, B_inds, B_len, ...
        ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput, ...
        itiRatesDiffInput, itiSuccPrecedInput, itiFailPrecedInput, rateSuccPrecedInput, rateFailPrecedInput,...
        vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds, vec_ratesdiff2, ratesDiffVecInds2] = ...
    trialHist_logistRegress...
        (alldata, doiti, trials_per_session, binRates, binITIs, uncommittedResp, allowCorrectResp, ...
        doplots, vec_ratesdiff, vec_iti, excludeExtraStim, excludeShortWaitDur, regressModel, vec_ratesdiff2, mouse);
    
    
    %%
    if ~isempty(B)
        sess_num = sess_num+1;
        dev_all(icount) = deviance;
        B_all(icount, 1:length(B)) = B;
        stats_all(icount) = stats;
    else
        [stats_all(icount).beta] = deal(NaN(length(B),1)); % again we go with a big estimate for the number of coefficients.
        [stats_all(icount).covb] = deal(NaN(length(B),length(B)));
        [stats_all(icount).se] = deal(NaN(length(B),1));
        [stats_all(icount).coeffcorr] = deal(NaN(length(B),length(B)));
        [stats_all(icount).t] = deal(NaN(length(B),1));
        [stats_all(icount).p] = deal(NaN(length(B),1));
    end
    
    
    %% Do conventional analysis to look at the effect of trial history for different ITIs.
    
    fprintf('Starting conventional analysis...\n')
    clear fract_change_choosingSameChoice_aftS fract_change_choosingSameChoice_aftF fract_change_choosingHR_aftHR_vs_LR_S fract_change_choosingLR_aftLR_vs_HR_S fract_change_choosingHR_aftHR_vs_LR_F fract_change_choosingLR_aftLR_vs_HR_F
    %     trialHist_post1
    if ~isempty(B) % if glmfit didn't work, it's usually due to small number of trials for a condition, so we don't run the conventional analysis as well!
        
        [fract_change_choosingSameChoice_aftS, fract_change_choosingSameChoice_aftF, ...
            fract_change_choosingHR_aftHR_vs_LR_S, fract_change_choosingLR_aftLR_vs_HR_S, ...
            fract_change_choosingHR_aftHR_vs_LR_F, fract_change_choosingLR_aftLR_vs_HR_F] = ...
            trialHist_0(...
            y, successPrevInput, failurePrevInput, itiPrecedInput, alldata, doiti, doplots_conventAnalysis, binningRates, vec_ratesdiff2, mouse);
        
        
        %%
        if strcmp(regressModel, 'rate_outcomeITI')
            fract_change_choosingSameChoice_aftS_all(icount,:) = fract_change_choosingSameChoice_aftS;
            fract_change_choosingSameChoice_aftF_all(icount,:) = fract_change_choosingSameChoice_aftF;
            fract_change_choosingHR_aftHR_vs_LR_S_all(icount,:) = fract_change_choosingHR_aftHR_vs_LR_S;
            fract_change_choosingLR_aftLR_vs_HR_S_all(icount,:) = fract_change_choosingLR_aftLR_vs_HR_S;
            fract_change_choosingHR_aftHR_vs_LR_F_all(icount,:) = fract_change_choosingHR_aftHR_vs_LR_F;
            fract_change_choosingLR_aftLR_vs_HR_F_all(icount,:) = fract_change_choosingLR_aftLR_vs_HR_F;
        else
            fract_change_choosingSameChoice_aftS_all =  [fract_change_choosingSameChoice_aftS_all; fract_change_choosingSameChoice_aftS];
            fract_change_choosingSameChoice_aftF_all = [fract_change_choosingSameChoice_aftF_all; fract_change_choosingSameChoice_aftF];
            fract_change_choosingHR_aftHR_vs_LR_S_all = [fract_change_choosingHR_aftHR_vs_LR_S_all; fract_change_choosingHR_aftHR_vs_LR_S];
            fract_change_choosingLR_aftLR_vs_HR_S_all = [fract_change_choosingLR_aftLR_vs_HR_S_all; fract_change_choosingLR_aftLR_vs_HR_S];
            fract_change_choosingHR_aftHR_vs_LR_F_all = [fract_change_choosingHR_aftHR_vs_LR_F_all; fract_change_choosingHR_aftHR_vs_LR_F];
            fract_change_choosingLR_aftLR_vs_HR_F_all = [fract_change_choosingLR_aftLR_vs_HR_F_all; fract_change_choosingLR_aftLR_vs_HR_F];
        end
        
    end
    
end


%%
disp('-------------------------- Done --------------------------')
fprintf(['Trials per mouse:', repmat('%d  ', [1, length(trials_per_mouse)]),'\n'], trials_per_mouse)
fprintf('Average trials per mouse: %d\n', nanmean(trials_per_mouse))


%% Set B, p and se for each term (rate, iti, etc) separately in a cell array

trialHist_post2


%% Set combined plots

if length(miceNames)>1
    trialHist_plots
end





%%
% B_all = NaN(length(miceNames), 10); % size of this depends on doiti and binRates and binITIs... needs improvement.
% B_overalbias_all = NaN(length(miceNames), 1);
% B_rate_all = NaN(length(miceNames), length(vec_ratesdiff));
% B_itiS_all = NaN(length(miceNames), size(vec_iti,2)-1);
% B_itiF_all = NaN(length(miceNames), size(vec_iti,2)-1);
% p_rate_all = NaN(length(miceNames), length(vec_ratesdiff));
% p_itiS_all = NaN(length(miceNames), size(vec_iti,2)-1);
% p_itiF_all = NaN(length(miceNames), size(vec_iti,2)-1);

%{
        if ~all([binITIs, binRates])
            B_all(icount, 1:length(B))  = B;
            
        elseif all([binITIs, binRates])
            % overal bias
            B_overalbias_all(icount) = B(1);
            
            
            % stim rate
            b = 2;
            e = b + length(ratesDiffVecInds)-1;
            B_rate_all(icount, ismember(vec_ratesdiff_all, vec_ratesdiff(ratesDiffVecInds))) = B(b:e);
            p_rate_all(icount, ismember(vec_ratesdiff_all, vec_ratesdiff(ratesDiffVecInds))) = stats.p(b:e);
            
            
            % itiS, itiF
            b = 1 + length(ratesDiffVecInds) +1;
            e = b + length(itiVecInds)-1;
            B_itiS_all(icount, :) = B(b:e);
            B_itiF_all(icount, :) = B(e+1:end);
            p_itiS_all(icount, :) = stats.p(b:e);
            p_itiF_all(icount, :) = stats.p(e+1:end);
        end
%}



%         fract_change_choosingSameChoice_aftS_all(icount, :) =  fract_change_choosingSameChoice_aftS;
%         fract_change_choosingSameChoice_aftF_all(icount, :) = fract_change_choosingSameChoice_aftF;
%         fract_change_choosingHR_aftHR_vs_LR_S_all(icount, :) = fract_change_choosingHR_aftHR_vs_LR_S;
%         fract_change_choosingLR_aftLR_vs_HR_S_all(icount, :) = fract_change_choosingLR_aftLR_vs_HR_S;
%         fract_change_choosingHR_aftHR_vs_LR_F_all(icount, :) = fract_change_choosingHR_aftHR_vs_LR_F;
%         fract_change_choosingLR_aftLR_vs_HR_F_all(icount, :) = fract_change_choosingLR_aftLR_vs_HR_F;