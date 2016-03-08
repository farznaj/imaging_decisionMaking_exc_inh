% analyze sessions of a mouse separately and then make an average across
% sessions (as opposed to pooling sessions, for which use trialHist_pooled).
miceNames = {'fn03'}; % {'fn03', 'fn04', 'fn05', 'fn06', 'fni16', 'fni17', 'fni18', 'fni19', 'hni01', 'hni04'};

binningRates = 1; % this is for the conventional analysis, if true the effect of outcome will be shown for different stim strength.

doplots = true;

regressModel = 'rateITI_outcomeRate'; % 'rate_outcomeITI' (default), 'rate_outcomeRate', 'rateITI_outcomeITI', 'rateITI_outcomeRate'
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
mouse = miceNames{1};
[day, dayLast, days2exclude] = setMouseAnalysisDays(mouse); % if u want to analyze only a particular session, define it and then run [alldata_fileNames, days_all] = setBehavFileNames(mouse, day);

[alldata_fileNames, days_all] = setBehavFileNames(mouse, day, dayLast, days2exclude);
fprintf('Total number of session: %d\n', length(alldata_fileNames))

[alldata, trials_per_session] = loadBehavData(alldata_fileNames, defaultHelpedTrs, saveHelpedTrs);
fprintf('Total number of trials: %d\n', length(alldata))
trials_per_mouse(1) = length(alldata);

begTrs = [0 cumsum(trials_per_session)]+1;
begTrs = begTrs(1:end-1);


%%
for icount = 1:length(trials_per_session)-1
    
    if trials_per_session(icount) > 20
        
        % set alldata for each session.
        [~,f] = fileparts(alldata_fileNames{icount});
        fprintf('-------------- session %d, %s --------------\n', icount, f)
        
        r = begTrs(icount) : begTrs(icount+1)-1;
        alldata_se = alldata(r);
        
        
        %% do logisitic regression
        clear B deviance stats X y B_inds B_len ratesDiffInput successPrevInput failurePrevInput itiPrecedInput
        
        [B, deviance, stats, X, y, B_inds, B_len, ...
            ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput, ...
            itiRatesDiffInput, itiSuccPrecedInput, itiFailPrecedInput, rateSuccPrecedInput, rateFailPrecedInput,...
            vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds, vec_ratesdiff2, ratesDiffVecInds2] = ...
            trialHist_logistRegress...
            (alldata_se, doiti, [], binRates, binITIs, uncommittedResp, allowCorrectResp, ...
            doplots, vec_ratesdiff, vec_iti, excludeExtraStim, excludeShortWaitDur, regressModel, vec_ratesdiff2, mouse);
        
        
        %%
        trialHist_post1

    end
end


%% set B, p and se for each term (rate, iti, etc) separately in a cell array
trialHist_post2


%% set combined plots
% if length(miceNames)>1
    trialHist_plots
% end







%%
%{
        if ~isempty(B)
            sess_num = sess_num+1;
            dev_all(icount) = deviance;
            B_all(icount, :) = B;
            stats_all(icount) = stats;           
            
            
            %% do the conventional analysis to look at the effect of trial history for different ITIs.
            [fract_change_choosingSameChoice_aftS, fract_change_choosingSameChoice_aftF, ...
                fract_change_choosingHR_aftHR_vs_LR_S, fract_change_choosingLR_aftLR_vs_HR_S, ...
                fract_change_choosingHR_aftHR_vs_LR_F, fract_change_choosingLR_aftLR_vs_HR_F] = trialHist_0(...
                y, successPrevInput, failurePrevInput, itiPrecedInput, alldata, doiti, doplots, binningRates, vec_ratesdiff2);
            
            
            %%
            fract_change_choosingSameChoice_aftS_all =  [fract_change_choosingSameChoice_aftS_all; fract_change_choosingSameChoice_aftS];
            fract_change_choosingSameChoice_aftF_all = [fract_change_choosingSameChoice_aftF_all; fract_change_choosingSameChoice_aftF];
            fract_change_choosingHR_aftHR_vs_LR_S_all = [fract_change_choosingHR_aftHR_vs_LR_S_all; fract_change_choosingHR_aftHR_vs_LR_S];
            fract_change_choosingLR_aftLR_vs_HR_S_all = [fract_change_choosingLR_aftLR_vs_HR_S_all; fract_change_choosingLR_aftLR_vs_HR_S];
            fract_change_choosingHR_aftHR_vs_LR_F_all = [fract_change_choosingHR_aftHR_vs_LR_F_all; fract_change_choosingHR_aftHR_vs_LR_F];
            fract_change_choosingLR_aftLR_vs_HR_F_all = [fract_change_choosingLR_aftLR_vs_HR_F_all; fract_change_choosingLR_aftLR_vs_HR_F];            
        end
%}


%{
css = [0 cumsum(B_len)];

B_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    B_term{ib} = B_all(:, css(ib)+1:css(ib+1));
end

sea = [stats_all.se]';
se_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    se_term{ib} = sea(:, css(ib)+1:css(ib+1));
end

pa = [stats_all.p]';
p_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    p_term{ib} = pa(:, css(ib)+1:css(ib+1));
end
%}