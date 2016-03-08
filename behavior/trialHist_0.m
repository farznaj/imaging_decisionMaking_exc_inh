function [fract_change_choosingSameChoice_aftS, fract_change_choosingSameChoice_aftF, ...
    fract_change_choosingHR_aftHR_vs_LR_S, fract_change_choosingLR_aftLR_vs_HR_S, ...
    fract_change_choosingHR_aftHR_vs_LR_F, fract_change_choosingLR_aftLR_vs_HR_F] = trialHist_0(...
    y, successPrevInput, failurePrevInput, itiPrecedInput, alldata, doiti, doplots, binningRates, vec_ratesdiff, mouse)


% compare fraction of choices to each side among different previous choices/outcomes.
% convential analysis to assess the results of regression model.

% run trialHist_logistRegress to get the vars you need here.


%% do logistic regression on pooled trials from all sessions
%{
doiti = true;
binRates = true;
binITIs = true; % false; %true;
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';
allowCorrectResp = 'change';
doplots = false;
excludeExtraStim = false;
vec_iti = [0 10 30]; % [];

% [B, deviance, stats, X, y] = trialHist_logistRegress(alldata, doiti, trials_per_session, binRates, binITIs, uncommittedResp, allowCorrectResp, doplots);

[B, deviance, stats, X, y, vec_ratesdiff, ratesDiffVecInds, vec_iti, itiVecInds,...
    ratesDiffInput, successPrevInput, failurePrevInput, itiPrecedInput] = trialHist_logistRegress...
    (alldata, doiti, trials_per_session, binRates, binITIs, uncommittedResp, allowCorrectResp, doplots, vec_iti, excludeExtraStim);
%}

%%
% binningRates = false;
norms = 'count'; % 'probability';


%%
if binningRates
    % if you want to control for the stim rate and compare choices for each bin separately use:
    v = vec_ratesdiff;
else
    % otherwise use something like [0 20] to get only 1 bin. (ie to not bin the data).
    v = [0 20];
end

[~, ~, stimrate] = setStimRateType(alldata);

stimrate(isnan(y)) = nan;

cb = unique([alldata.categoryBoundaryHz]); % categ boundary in hz


%%%%%%%%%%%%%%%%%%%%%%%%% Preceded by success %%%%%%%%%%%%%%%%%%%%%%%%%

%% trials preceded by a previous successful HR choice.

f = (successPrevInput==1);

choiceNow_all = y(f); % choice on trials preceded by a previous successful HR choice.
stimdiffNow_all = stimrate(f)-cb; % stim rate on trials preceded by a previous successful HR choice.
if doiti
    t = itiPrecedInput(f,:);
end
% r = ratesDiffInput(f,:); % instead of binning stimrate here u can use ratesDiffInput
[nhs,edg,bin] = histcounts(abs(stimdiffNow_all), v, 'normalization', norms);
allbins = unique(bin);
allbins(~allbins) = [];
Nchoice_Nstim_HR_LR_aftHR_S = NaN(length(allbins), size(t,2));
Nchoice_Nstim_HR_aftHR_S = NaN(length(allbins), size(t,2));
Nchoice_Nstim_LR_aftHR_S = NaN(length(allbins), size(t,2));
st = sum(t==1);
choiceNow = NaN(max(st), size(t,2));
stimdiffNow = NaN(max(st), size(t,2));

for ib = allbins'
    if ~doiti
        choiceNow = choiceNow_all(bin==ib);
        stimdiffNow = stimdiffNow_all(bin==ib);
    else
        for it = 1:size(t,2)
            a = (bin==ib & t(:,it)==1);
            if sum(a)>3 % we want at least 3 trials.
                choiceNow(1:sum(a), it) = choiceNow_all(a);
                stimdiffNow(1:sum(a), it) = stimdiffNow_all(a);
            end
        end
        %{
        choiceNow_t1 = choiceNow_all(bin==ib & t(:,1)==1);
        choiceNow_t2 = choiceNow_all(bin==ib & t(:,2)==1);
        stimdiffNow_t1 = stimdiffNow_all(bin==ib & t(:,1)==1);
        stimdiffNow_t2 = stimdiffNow_all(bin==ib & t(:,2)==1);
        choiceNow = padcat(choiceNow_t1, choiceNow_t2);
        stimdiffNow = padcat(stimdiffNow_t1, stimdiffNow_t2);
        %}
    end
    
    choice_HR_LR = sum(choiceNow==1) ./ sum(choiceNow==0);
    stim_HR_LR = sum(sign(stimdiffNow)==1) ./ sum(sign(stimdiffNow)==-1);
    
    Nchoice_Nstim_HR_LR_aftHR_S(ib,:) = choice_HR_LR ./ stim_HR_LR;
    % choice_HR_LR - stim_HR_LR; % not a good measure for bias due to previous trial; this is negative so
    % after a HR success trial there is little bias to LR choice! ... the
    % effect of overall bias is more than the effect of the previous-trials
    % bias.
    
    Nchoice_Nstim_HR_aftHR_S(ib,:) = sum(choiceNow==1) ./ sum(sign(stimdiffNow)==1);
    Nchoice_Nstim_LR_aftHR_S(ib,:) = sum(choiceNow==0) ./ sum(sign(stimdiffNow)==-1);
end




%% trials preceded by a previous successful LR choice.

f = (successPrevInput==-1);

choiceNow_all = y(f); % choice on trials preceded by a previous successful HR choice.
stimdiffNow_all = stimrate(f)-cb; % stim rate on trials preceded by a previous successful HR choice.
if doiti
    t = itiPrecedInput(f,:);
end

[nls,edg,bin] = histcounts(abs(stimdiffNow_all), v, 'normalization', norms);
allbins = unique(bin);
allbins(~allbins) = [];
Nchoice_Nstim_HR_LR_aftLR_S = NaN(length(allbins), size(t,2));
Nchoice_Nstim_LR_aftLR_S = NaN(length(allbins), size(t,2));
Nchoice_Nstim_HR_aftLR_S = NaN(length(allbins), size(t,2));
st = sum(t==1);
choiceNow = NaN(max(st), size(t,2));
stimdiffNow = NaN(max(st), size(t,2));

for ib = allbins'
    if ~doiti
        choiceNow = choiceNow_all(bin==ib);
        stimdiffNow = stimdiffNow_all(bin==ib);
    else
        for it = 1:size(t,2)
            a = (bin==ib & t(:,it)==1);
            if sum(a)>3 % we want at least 3 trials.
                choiceNow(1:sum(a), it) = choiceNow_all(a);
                stimdiffNow(1:sum(a), it) = stimdiffNow_all(a);
            end
        end
    end
    
    choice_HR_LR = sum(choiceNow==1) ./ sum(choiceNow==0);
    stim_HR_LR = sum(sign(stimdiffNow)==1) ./ sum(sign(stimdiffNow)==-1);
    
    Nchoice_Nstim_HR_LR_aftLR_S(ib,:) = choice_HR_LR ./ stim_HR_LR;
    % choice_LR_HR - stim_LR_HR; % if >0 mouse went more to LR after a LR success than expected from the stim rate.
    % after a LR success trial there is large bias toward LR choice ... but
    % this is bc of this animal's overal LR bias.
    
    Nchoice_Nstim_LR_aftLR_S(ib,:) = sum(choiceNow==0) ./ sum(sign(stimdiffNow)==-1);
    Nchoice_Nstim_HR_aftLR_S(ib,:) = sum(choiceNow==1) ./ sum(sign(stimdiffNow)==1);
end





%%%%%%%%%%%%%%%%%%%%%%%%% Preceded by failure %%%%%%%%%%%%%%%%%%%%%%%%%

%% trials preceded by a previous failure HR choice.

f = (failurePrevInput==1);

choiceNow_all = y(f); % choice on trials preceded by a previous failure HR choice.
stimdiffNow_all = stimrate(f)-cb; % stim rate on trials preceded by a previous failure HR choice.
if doiti
    t = itiPrecedInput(f,:);
end

[nhf,edg,bin] = histcounts(abs(stimdiffNow_all), v, 'normalization', norms);
allbins = unique(bin);
allbins(~allbins) = [];
Nchoice_Nstim_HR_LR_aftHR_F = NaN(length(allbins), size(t,2));
Nchoice_Nstim_HR_aftHR_F = NaN(length(allbins), size(t,2));
Nchoice_Nstim_LR_aftHR_F = NaN(length(allbins), size(t,2));
st = sum(t==1);
choiceNow = NaN(max(st), size(t,2));
stimdiffNow = NaN(max(st), size(t,2));

for ib = allbins'
    if ~doiti
        choiceNow = choiceNow_all(bin==ib);
        stimdiffNow = stimdiffNow_all(bin==ib);
    else
        for it = 1:size(t,2)
            a = (bin==ib & t(:,it)==1);
            if sum(a)>3 % we want at least 3 trials.
                choiceNow(1:sum(a), it) = choiceNow_all(a);
                stimdiffNow(1:sum(a), it) = stimdiffNow_all(a);
            end
        end
    end
    
    choice_HR_LR = sum(choiceNow==1) ./ sum(choiceNow==0);
    stim_HR_LR = sum(sign(stimdiffNow)==1) ./ sum(sign(stimdiffNow)==-1);
    
    Nchoice_Nstim_HR_LR_aftHR_F(ib,:) = choice_HR_LR ./ stim_HR_LR;
    % choice_HR_LR - stim_HR_LR;
    % after a HR failure trial there is little bias to HR choice!
    
    Nchoice_Nstim_HR_aftHR_F(ib,:) = sum(choiceNow==1) ./ sum(sign(stimdiffNow)==1);
    Nchoice_Nstim_LR_aftHR_F(ib,:) = sum(choiceNow==0) ./ sum(sign(stimdiffNow)==-1);
end


%% trials preceded by a previous failure LR choice.

f = (failurePrevInput==-1);

choiceNow_all = y(f); % choice on trials preceded by a previous failure HR choice.
stimdiffNow_all = stimrate(f)-cb; % stim rate on trials preceded by a previous failure HR choice.
if doiti
    t = itiPrecedInput(f,:);
end

[nlf,edg,bin] = histcounts(abs(stimdiffNow_all), v, 'normalization', norms);
allbins = unique(bin);
allbins(~allbins) = [];
Nchoice_Nstim_HR_LR_aftLR_F = NaN(length(allbins), size(t,2));
Nchoice_Nstim_LR_aftLR_F = NaN(length(allbins), size(t,2));
Nchoice_Nstim_HR_aftLR_F = NaN(length(allbins), size(t,2));
st = sum(t==1);
choiceNow = NaN(max(st), size(t,2));
stimdiffNow = NaN(max(st), size(t,2));

for ib = allbins'
    if ~doiti
        choiceNow = choiceNow_all(bin==ib);
        stimdiffNow = stimdiffNow_all(bin==ib);
    else
        for it = 1:size(t,2)
            a = (bin==ib & t(:,it)==1);
            if sum(a)>3 % we want at least 3 trials.
                choiceNow(1:sum(a), it) = choiceNow_all(a);
                stimdiffNow(1:sum(a), it) = stimdiffNow_all(a);
            end
        end
    end
    
    choice_HR_LR = sum(choiceNow==1) ./ sum(choiceNow==0);
    stim_HR_LR = sum(sign(stimdiffNow)==1) ./ sum(sign(stimdiffNow)==-1);
    
    Nchoice_Nstim_HR_LR_aftLR_F(ib,:) = choice_HR_LR ./ stim_HR_LR;
    % choice_LR_HR - stim_LR_HR;
    % after a LR failure trial there is a large bias toward LR choice ... but
    % this is bc of this animal's overal LR bias.
    
    Nchoice_Nstim_LR_aftLR_F(ib,:) = sum(choiceNow==0) ./ sum(sign(stimdiffNow)==-1);
    Nchoice_Nstim_HR_aftLR_F(ib,:) = sum(choiceNow==1) ./ sum(sign(stimdiffNow)==1);
end


%% check number of trials in each bin (or if no bins, then number of trials
% for each hs ls hf lf category)
if doplots
    figure('name', ['Mouse ', mouse]);
    subplot(311)
    plot([nhs;nls;nhf;nlf]')
end


%% compare effect of previous choice (combined measure for L and R)

% succ
s = [Nchoice_Nstim_HR_LR_aftHR_S;  Nchoice_Nstim_HR_LR_aftLR_S];
fract_change_choosingSameChoice_aftS = Nchoice_Nstim_HR_LR_aftHR_S ./ Nchoice_Nstim_HR_LR_aftLR_S; % s(1,:)./s(2,:);
% fail
f = [Nchoice_Nstim_HR_LR_aftHR_F;  Nchoice_Nstim_HR_LR_aftLR_F];
fract_change_choosingSameChoice_aftF = Nchoice_Nstim_HR_LR_aftHR_F ./ Nchoice_Nstim_HR_LR_aftLR_F; % f(1,:)./f(2,:);

% if length(fract_change_choosingSameChoice_aftS)>1
%     aveS_aveF = [nanmean(fract_change_choosingSameChoice_aftS)  nanmean(fract_change_choosingSameChoice_aftF)]
% end

if doplots
%     figure;
    subplot(312)
    bar([fract_change_choosingSameChoice_aftS; fract_change_choosingSameChoice_aftF])
    set(gca, 'xticklabel', {'S', 'F'})
end


%% compare effect of previous choice (separate measures for L and R)

%%% if the values below are >0 then there is bias due to previous choice (successful)
fract_change_choosingHR_aftHR_vs_LR_S = Nchoice_Nstim_HR_aftHR_S ./ Nchoice_Nstim_HR_aftLR_S; % >0 : animal goes more to HR after HR success.
fract_change_choosingLR_aftLR_vs_HR_S = Nchoice_Nstim_LR_aftLR_S ./ Nchoice_Nstim_LR_aftHR_S; % >0 : animal goes more to LR after LR success.

% if length(fract_change_choosingSameChoice_aftS)>1
%     aveHrS_aveLrS = [nanmean(fract_change_choosingHR_aftHR_vs_LR_S), nanmean(fract_change_choosingLR_aftLR_vs_HR_S)]
% end
% mouse chose more HR after a HR success trial compared to a LR success trial.
% mouse chose more LR after a LR success trial compared to a HR success tr.


%%% if the values below are both >0 then there is bias due to previous failure trial.
fract_change_choosingHR_aftHR_vs_LR_F = Nchoice_Nstim_HR_aftHR_F ./ Nchoice_Nstim_HR_aftLR_F; % >0 : animal goes more to HR after HR success.
fract_change_choosingLR_aftLR_vs_HR_F = Nchoice_Nstim_LR_aftLR_F ./ Nchoice_Nstim_LR_aftHR_F; % >0 : animal goes more to LR after LR success.

% if length(fract_change_choosingSameChoice_aftS)>1
%     aveHrF_aveLrF = [nanmean(fract_change_choosingHR_aftHR_vs_LR_F), nanmean(fract_change_choosingLR_aftLR_vs_HR_F)]
% end


if doplots
    subplot(313)
    bar([fract_change_choosingHR_aftHR_vs_LR_S; fract_change_choosingLR_aftLR_vs_HR_S;...
        fract_change_choosingHR_aftHR_vs_LR_F; fract_change_choosingLR_aftLR_vs_HR_F])
    set(gca, 'xticklabel', {'S-HR', 'S-LR', 'F-HR', 'F-LR'})
end


%% compare effect of previous outcome (s,f)
%{
fract_change_choosingHR_aftS_vs_aftF_1 = Nchoice_Nstim_HR_LR_aftHR_S ./ Nchoice_Nstim_HR_LR_aftHR_F
fract_change_choosingLR_aftS_vs_aftF_1 = (1./Nchoice_Nstim_HR_LR_aftLR_S) ./ (1./Nchoice_Nstim_HR_LR_aftLR_F)

fract_change_choosingHR_aftS_vs_aftF_2 = Nchoice_Nstim_HR_aftHR_S ./ Nchoice_Nstim_HR_aftHR_F
fract_change_choosingLR_aftS_vs_aftF_2 = Nchoice_Nstim_LR_aftLR_S ./ Nchoice_Nstim_LR_aftLR_F


fract_change_choosingSameChoice_aftS_vs_aftF = fract_change_choosingSameChoice_aftS ./ fract_change_choosingSameChoice_aftF
%}


%%
%{
% - find trials that have the same sensory stimulus (eg. left-indicating)
%      - classify them based on their choice history:
%           eg. correct left vs. correct right
%           - long vs short ITI

%%
figure; plot(sort(abs(stimrate-cb)))
sum((stimrate-cb) >= 8);
trsnow = (stimrate-cb) >= 8;
figure; plot(outcomes(trsnow))
alldata(1).highRateChoicePort

%%
trback = 1;
trs = find(trsnow);
trs(trs==1) = []; % b

trsb = trs-trback;
trsb(trsb<1) = [];

out_trsb = outcomes(trsb);
resp_trsb = allResp(trsb); %1: left, 2: right
%}
