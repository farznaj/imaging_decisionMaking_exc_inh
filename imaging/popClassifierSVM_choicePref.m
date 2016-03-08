% see if the classifer weights are the ROC-drived choicePref are some how
% similar.

% Compare sign of svmBeta with choicePref.

%% set ipsi and contra trials
thStimStrength = 0; % 2; % what stim strength you want to use for computing choice pref.
useEqualNumTrs = 0; true; % if true, equal number of trials for HR and LR will be used to compute ROC.


correctL = (outcomes==1) & (allResp==1);
correctR = (outcomes==1) & (allResp==2);

ipsiTrs = (correctL' &  abs(stimrate-cb) > thStimStrength);
contraTrs = (correctR' &  abs(stimrate-cb) > thStimStrength);
fprintf('Num corrL and corrR (stim diff > %d): %d,  %d\n', [thStimStrength, sum(ipsiTrs) sum(contraTrs)])

makeplots = 0;


%%
% compute choicePref for each frame
% choicePref_all: frames x units. chiocePref at each frame for each neuron
% choicePref_all = choicePref_ROC(traces_al_sm, ipsiTrs, contraTrs, makeplots, eventI_stimOn);

% traces_al_sm_aveFr = nanmean(traces_al_sm(eventI_stimOn:end,:,:), 1); % compute choicePref for the average of frames after the stim.

traces_al_sm_aveFr = nanmean(traces_al_sm(ep, ~NsExcluded,:), 1); % same neurons and epoch that were analyzed for SVM.

choicePref_all = choicePref_ROC(traces_al_sm_aveFr, ipsiTrs, contraTrs, makeplots, eventI_stimOn, useEqualNumTrs); % frames x neurons

% choicePrefAveEp = nanmean(choicePref_all(ep, :)); % average choicePref across epoch ep.
choicePrefAveEp = choicePref_all;

% choicePrefAveEp(NsFewTrActiv) = []; % use same set of neurons that went into SVM analysis.


%% Compare svmBeta with choicePref.
b = wNsHrLrAve;
c = corrcoef(-choicePrefAveEp, b);

figure; 
subplot(2,2,[1,2]), hold on
plot(b)
plot(-choicePrefAveEp) % you plot negative of that bc you've assigned ipsi - and contra + in the choicePref code. but in SVM u've assigned HR + and LR -. For this mouse HR is ipsi.... u need to do this properly.
plot([1 length(b)], [0 0], 'handlevisibility', 'off')
legend('svmBeta', 'ROC choicePref')
xlabel('Neuron')
ylabel('LR <--    --> HR');
title(sprintf('CorrCoef = %.3f', c(2)))
 
aa = b(-choicePrefAveEp > 0);
bb = b(-choicePrefAveEp < 0);
[nanmean(aa>0),... % fraction of trs w HR (ipsi) choice pref that have positive svm beta
nanmean(bb<0)] % fraction of trs w LR (contra) choice pref that have negative svm beta

%{
[s,i] = sort(-choicePrefAveEp);
subplot(212); hold on
plot(s)
plot(b(i))
legend('choicePref', 'svmBeta')
%}


% compare classifier weights for neurons with higher chioce preferance vs
% lower choice preference.
[~, i] = sort(abs(choicePrefAveEp));
q = floor(length(choicePrefAveEp)/4);
x1 = abs(wNsHrLrAve(i(1:q))); % classifier weights of neurons with low choice prefereance
x2 = abs(wNsHrLrAve(i(end-q+1:end)));% .. with high choice preferance.

subplot(223), hold on
[n1, ed1] = histcounts(x1, 'normalization', 'probability');
[n2, ed2] = histcounts(x2, 'normalization', 'probability');
plot(ed1(1:end-1), n1)
plot(ed2(1:end-1), n2)
xlabel('Classifer weight')
ylabel('Fraction of neurons')
legend('Low choicePref', 'High choicePref')

subplot(224), hold on 
a = nanmean([x1 x2]);
s = nanstd([x1 x2]);
errorbar(1:2, a,s)
xlabel('Choice preferance')
set(gca, 'xtick', [1,2])
set(gca, 'xticklabel', {'low', 'high'})
ylabel('Classifier weight')

