% this is an old version. use trialHist_pooled and
% trialHist_1mouse_eachSession instead.

% for a single mouse use this code
% for several mice use : trialHist_micePooled

%%
mouse = 'hni01';

%%
[day, dayLast, days2exclude] = setMouseAnalysisDays(mouse);

defaultHelpedTrs = false; % set to 0 if you want to manually set the helped trials.
saveHelpedTrs = true;


%%
[alldata_fileNames, days_all] = setBehavFileNames(mouse, day, dayLast, days2exclude);

% showcell(alldata_fileNames')
showcell(days_all')
fprintf('Total number of session: %d\n', length(alldata_fileNames))
% length(days_all)


%% load alldata files
[alldata, trials_per_session] = loadBehavData(alldata_fileNames, defaultHelpedTrs, saveHelpedTrs);

fprintf('Total number of trials: %d\n', length(alldata))
% sum(trials_per_session)


%% do logistic regression on pooled trials from all sessions
doiti = true;
binRates = true;
binITIs = true; % false; %true;
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';
allowCorrectResp = 'change';
doplots = true;
excludeExtraStim = false;
excludeShortWaitDur = true;
vec_iti = [0 7 30]; % [0 6 9 12 30]; % [0 7 30]; % [0 10 30]; % [0 6 9 12 30];
regressModel = 'rateITI_outcomeITI'; % 'rate_outcomeITI' (default), 'rate_outcomeRate', 'rateITI_outcomeITI', 'rateITI_outcomeRate' 

% [B, deviance, stats, X, y] = trialHist_logistRegress(alldata, doiti, trials_per_session, binRates, binITIs, uncommittedResp, allowCorrectResp, doplots);

[B, deviance, stats, X, y, vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds,...
    ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput] = trialHist_logistRegress...
    (alldata, doiti, trials_per_session, binRates, binITIs, uncommittedResp, allowCorrectResp, doplots, vec_iti, excludeExtraStim, excludeShortWaitDur, regressModel);


%% do the conventional analysis to look at the effect of trial history for different ITIs.
[fract_change_choosingSameChoice_aftS, fract_change_choosingSameChoice_aftF, ...
    fract_change_choosingHR_aftHR_vs_LR_S, fract_change_choosingLR_aftLR_vs_HR_S, ...
    fract_change_choosingHR_aftHR_vs_LR_F, fract_change_choosingLR_aftLR_vs_HR_F] = trialHist_0(...
    y, successPrevInput, failurePrevInput, itiPrecedInput, alldata, doiti, doplots, vec_ratesdiff);


%% do logistic regression on each session separately
%{
doiti = true;
binRates = true;
binITIs = true; % false; %true;
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';
allowCorrectResp = 'change';
excludeExtraStim = false;
vec_iti = [0 10 30]; % [0 6 9 12 30]; %
%}
doplots = false;
begTrs = [0 cumsum(trials_per_session)]+1;
begTrs = begTrs(1:end-1);

vec_ratesdiff_all = 0:2:12; %vec_ratesdiff(1:end-1);
vec_iti_all = vec_iti; % [0 10 30]; % [0 7 30]; % [0 6 9 12 30] % this needs improvement. u r using the same vec in the regress code, so maybe input it ...

B_all = NaN(length(trials_per_session), 10); % size of this determins on doiti and binRates and binITIs... needs improvement.
B_overalbias_all = NaN(length(trials_per_session), 1);
B_rate_all = NaN(length(trials_per_session), length(vec_ratesdiff_all));
B_itiS_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
B_itiF_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
p_rate_all = NaN(length(trials_per_session), length(vec_ratesdiff_all));
p_itiS_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
p_itiF_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
dev_all = NaN(1, length(trials_per_session));
clear stats_all
sess_num = 0;

fract_change_choosingSameChoice_aftS_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
fract_change_choosingSameChoice_aftF_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
fract_change_choosingHR_aftHR_vs_LR_S_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
fract_change_choosingLR_aftLR_vs_HR_S_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
fract_change_choosingHR_aftHR_vs_LR_F_all = NaN(length(trials_per_session), size(vec_iti,2)-1);
fract_change_choosingLR_aftLR_vs_HR_F_all = NaN(length(trials_per_session), size(vec_iti,2)-1);


for ise = 1:length(trials_per_session)-1
    
    if trials_per_session(ise) > 20
        
        % do logistic regression
        
        [~,f] = fileparts(alldata_fileNames{ise});
        fprintf('-------------- session %d, %s --------------\n', ise, f)
        
        r = begTrs(ise) : begTrs(ise+1)-1;
        alldata_se = alldata(r);
        
%         [B, deviance, stats, X, y, vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds] = trialHist_logistRegress(alldata_se, doiti, [], binRates, binITIs, uncommittedResp, allowCorrectResp, doplots, vec_iti, excludeExtraStim, excludeShortWaitDur);

        [B, deviance, stats, X, y, B_inds, B_len, ...
            ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput, ...
            vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds, vec_ratesdiff2, ratesDiffVecInds2] = trialHist_logistRegress...
            (alldata_se, doiti, [], binRates, binITIs, uncommittedResp, allowCorrectResp, ...
            doplots, vec_ratesdiff, vec_iti, excludeExtraStim, excludeShortWaitDur, regressModel, vec_ratesdiff2);
        
        
        %%
        if ~isempty(B)
            
            sess_num = sess_num+1;
            dev_all(ise) = deviance;
            stats_all(ise) = stats;
            
            if ~all([binITIs, binRates])
                B_all(ise, 1:length(B))  = B;
                
            elseif all([binITIs, binRates])
                % overal bias
                B_overalbias_all(ise) = B(1);
                
                
                % stim rate
                b = 2;
                e = b + length(ratesDiffVecInds)-1;
                B_rate_all(ise, ismember(vec_ratesdiff_all, vec_ratesdiff(ratesDiffVecInds))) = B(b:e);
                p_rate_all(ise, ismember(vec_ratesdiff_all, vec_ratesdiff(ratesDiffVecInds))) = stats.p(b:e);
                
                
                % itiS, itiF
                b = 1 + length(ratesDiffVecInds) +1;
                e = b + length(itiVecInds)-1;
                B_itiS_all(ise, :) = B(b:e);
                B_itiF_all(ise, :) = B(e+1:end);
                p_itiS_all(ise, :) = stats.p(b:e);
                p_itiF_all(ise, :) = stats.p(e+1:end);
            end
            
            
            %% do the conventional analysis to look at the effect of trial history for different ITIs.
            [fract_change_choosingSameChoice_aftS, fract_change_choosingSameChoice_aftF, ...
                fract_change_choosingHR_aftHR_vs_LR_S, fract_change_choosingLR_aftLR_vs_HR_S, ...
                fract_change_choosingHR_aftHR_vs_LR_F, fract_change_choosingLR_aftLR_vs_HR_F] = trialHist_0(...
                y, successPrevInput, failurePrevInput, itiPrecedInput, alldata, doiti, doplots, vec_ratesdiff);
            
            %%
            fract_change_choosingSameChoice_aftS_all(ise, :) =  fract_change_choosingSameChoice_aftS;
            fract_change_choosingSameChoice_aftF_all(ise, :) = fract_change_choosingSameChoice_aftF;
            fract_change_choosingHR_aftHR_vs_LR_S_all(ise, :) = fract_change_choosingHR_aftHR_vs_LR_S;
            fract_change_choosingLR_aftLR_vs_HR_S_all(ise, :) = fract_change_choosingLR_aftLR_vs_HR_S;
            fract_change_choosingHR_aftHR_vs_LR_F_all(ise, :) = fract_change_choosingHR_aftHR_vs_LR_F;
            fract_change_choosingLR_aftLR_vs_HR_F_all(ise, :) = fract_change_choosingLR_aftLR_vs_HR_F;
            
            
        end
    end
end

fprintf('Total number of sessions went into the analysis: %d\n', sess_num)


%% session by session: plots

trialHist_logistRegress_plots



%%