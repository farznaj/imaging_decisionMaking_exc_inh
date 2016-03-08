% it will pool all sessions for each mouse.

miceNames = {'fn03', 'fn04', 'fn05', 'fn06', 'fni16', 'fni17', 'fni18', 'fni19', 'hni01', 'hni04'};

binningRates = 1; % this is for the conventional analysis, if true the effect of outcome will be shown for different stim strength.

doplots = true;

regressModel = 'rate_outcomeITI'; % 'rate_outcomeITI' (default), 'rate_outcomeRate', 'rateITI_outcomeITI', 'rateITI_outcomeRate'
doiti = true;
binRates = true;
binITIs = true; % false; %true;

allowCorrectResp = 'change';
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';

excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;

vec_iti = [0 10 30]; %[0 6 9 12 30]; % [0 7 30]; % [0 10 30]; % [0 6 9 12 30]; % use [0 40]; if you want to have a single iti bin and in conventioinal analysis look at the effect of current rate on outcome.
vec_ratesdiff = 0:2:12;
vec_ratesdiff2 = [0 6 10 12];

defaultHelpedTrs = false; % false; % set to 0 if you want to manually set the helped trials.
saveHelpedTrs = true;


%%
dev_all = NaN(1, length(miceNames));
clear stats_all B_all
sess_num = 0;
fract_change_choosingSameChoice_aftS_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
fract_change_choosingSameChoice_aftF_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
fract_change_choosingHR_aftHR_vs_LR_S_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
fract_change_choosingLR_aftLR_vs_HR_S_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
fract_change_choosingHR_aftHR_vs_LR_F_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
fract_change_choosingLR_aftLR_vs_HR_F_all = []; % NaN(length(miceNames), size(vec_iti,2)-1);
trials_per_mouse = NaN(1, length(miceNames));


%%
for icount = 1:length(miceNames)
    
    fprintf('--------------------------\n')
    mouse = miceNames{icount};
    [day, dayLast, days2exclude] = setMouseAnalysisDays(mouse);
    
    [alldata_fileNames, days_all] = setBehavFileNames(mouse, day, dayLast, days2exclude);
    fprintf('Total number of session: %d\n', length(alldata_fileNames))
    
    [alldata, trials_per_session] = loadBehavData(alldata_fileNames, defaultHelpedTrs, saveHelpedTrs);
    fprintf('Total number of trials: %d\n', length(alldata))
    trials_per_mouse(icount) = length(alldata);
    
    
    %% do logisitic regression
    clear B deviance stats X y B_inds B_len ratesDiffInput successPrevInput failurePrevInput itiPrecedInput
    
    [B, deviance, stats, X, y, B_inds, B_len, ...
        ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput, ...
        itiRatesDiffInput, itiSuccPrecedInput, itiFailPrecedInput, rateSuccPrecedInput, rateFailPrecedInput,...
        vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds, vec_ratesdiff2, ratesDiffVecInds2] = ...
        trialHist_logistRegress...
        (alldata, doiti, trials_per_session, binRates, binITIs, uncommittedResp, allowCorrectResp, ...
        doplots, vec_ratesdiff, vec_iti, excludeExtraStim, excludeShortWaitDur, regressModel, vec_ratesdiff2, mouse);
    
    
    %%
    trialHist_post1

    
end

fprintf(['Trials per mouse:', repmat('%d  ', [1, length(trials_per_mouse)]),'\n'], trials_per_mouse)
fprintf('Average trials per mouse: %d\n', nanmean(trials_per_mouse))


%% set B, p and se for each term (rate, iti, etc) separately in a cell array
trialHist_post2


%% set combined plots
if length(miceNames)>1
    trialHist_plots
end





%%
% B_all = NaN(length(miceNames), 10); % size of this determins on doiti and binRates and binITIs... needs improvement.
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