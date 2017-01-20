saveVars = 1; % if 1, vars will be saved.
savefigs = 1;
trialHistAnalysis = 1;
mouse = 'fni16'; %'fni17';

thStimStrength = 0; % 2; % what stim strength you want to use for computing choice pref.
makeplots = 0;
useEqualNumTrs = 0; % if true, equal number of trials for HR and LR will be used to compute ROC.
frameLength = 1000/30.9; % sec.

if strcmp(mouse, 'fni17')
    ep_ms = [800, 1100]; % window to compute ROC distributions ... for fni17 you used [800 1100] to compute svm which does not include choice... this is not the case for fni16.
    
    days = {'151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};
    
elseif strcmp(mouse, 'fni16')
    % you use the ep of each day to compute ROC ave for setting ROC histograms.
    
    days = {'150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'};
    
end



%{
days = {'151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', ...
    '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', ...
    '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'};
%}


%%
ep_all = nan(length(days),2);

for nt = 0:1 % neuron type (0:exc ; 1:inh)    
    
    if nt==0
        disp('Analyzing excitatory neurons.')
        choicePref_all_alld_exc = cell(1, length(days));
        
    elseif nt==1
        disp('Analyzing inhibitory neurons.')
        choicePref_all_alld_inh = cell(1, length(days));
        
    end
    eventI_stimOn_all = nan(1,length(days));
    
    
    %%
    for iday = 1:length(days)
        
        disp('__________________________________________________________________')
        dn = simpleTokenize(days{iday}, '_');
        
        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
        
        fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})
        
        signalCh = 2; % because you get A from channel 2, I think this should be always 2.
        pnev2load = [];
        [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
        [pd, pnev_n] = fileparts(pnevFileName);
        moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
        postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
        
        
        if trialHistAnalysis==1
            load(postName, 'trialHistory','stimAl_allTrs')
            stimAl = stimAl_allTrs;
        else
            load(postName, 'outcomes', 'allResp_HR_LR', 'stimrate', 'cb','stimAl_noEarlyDec')
            stimAl = stimAl_noEarlyDec;
        end
        
        load(moreName, 'inhibitRois')
        
        
        %%
        load(postName, 'timeStimOnset', 'time1stSideTry')
        
        ep_ms0 = [floor(nanmin(time1stSideTry-timeStimOnset))-30-300, floor(nanmin(time1stSideTry-timeStimOnset))-30];
        fprintf('Training window: [%d %d] ms\n', ep_ms0(1), ep_ms0(2))
        
        ep_all(iday,:) = ep_ms0;
        
        
        %%
        traces_al_sm0 = stimAl.traces;
        eventI_stimOn = stimAl.eventI;
        
        
        %%
        traces_al_sm = traces_al_sm0(:,inhibitRois==nt,:);
        trsExcluded = isnan(squeeze(mean(mean(traces_al_sm,1),2)));
        
        
        %% Compute and plot choice preference, 2*(auc-0.5), for each neuron at each frame.
        
        % choicePref_ROC
        
        % set ipsi and contra trials
        if trialHistAnalysis==0
            correctL = (outcomes==1) & (allResp_HR_LR==1);
            correctR = (outcomes==1) & (allResp_HR_LR==0);
            
            ipsiTrs = (correctL')  &  abs(stimrate-cb) > thStimStrength;
            contraTrs = (correctR')  &  abs(stimrate-cb) > thStimStrength;
        else
            correctL = trialHistory.choiceVec0(:,3)==1;
            correctR = trialHistory.choiceVec0(:,3)==0;
            
            ipsiTrs = (correctL'); % &  abs(stimrate-cb) > thStimStrength);
            contraTrs = (correctR'); % &  abs(stimrate-cb) > thStimStrength);
        end
        
        fprintf('Num corrL and corrR (stim diff > %d): %d,  %d\n', [thStimStrength, sum(ipsiTrs) sum(contraTrs)])
        
        
        %%
        traces_al_sm = traces_al_sm(:,:,~trsExcluded);
        ipsiTrs = ipsiTrs(~trsExcluded);
        contraTrs = contraTrs(~trsExcluded);
        
        
        %%
        % compute choicePref for each frame
        % choicePref_all: frames x units. chiocePref at each frame for each neuron
        choicePref_all = choicePref_ROC(traces_al_sm, ipsiTrs, contraTrs, makeplots, eventI_stimOn, useEqualNumTrs);
        
        
        %%
        if nt==0
            choicePref_all_alld_exc{iday} = choicePref_all; % frames x neurons
        elseif nt==1
            choicePref_all_alld_inh{iday} = choicePref_all; % frames x neurons
        end
        
        eventI_stimOn_all(iday) = eventI_stimOn;
        
        
    end    
end

mean(ep_all)
min(ep_all)
max(ep_all)


%%
if saveVars
    cd(fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse))
    
    if trialHistAnalysis
        namv = sprintf('roc_prev_stimstr%d.mat', thStimStrength);
    else
        namv = sprintf('roc_curr_stimstr%d.mat', thStimStrength);
    end
    
    if exist(namv, 'file')==2
        error('File already exists... do you want to over-write it?')
    else
        disp('saving roc vars for exc and inh neurons....')
        save(namv, 'eventI_stimOn_all', 'choicePref_all_alld_exc', 'choicePref_all_alld_inh')
    end
end


%% Load and align all traces on stim onset
% run the 1st section of this script
% cd /home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC
% load('roc_curr.mat')
% load('roc_prev.mat')

nPost = nan(1,length(days));
for iday = 1:length(days)
    nPost(iday) = size(choicePref_all_alld_exc{iday},1) - eventI_stimOn_all(iday);
end

nPreMin = min(eventI_stimOn_all)-1;
nPostMin = min(nPost);

%%%
a = -frameLength * (0:nPreMin); a = a(end:-1:1);
b = frameLength * (1:nPostMin);
time_aligned = [a,b];

%%%
choicePref_exc_aligned = cell(1,length(days)); % nan(nPreMin + nPostMin + 1, length(days));
choicePref_inh_aligned = cell(1,length(days)); % nan(nPreMin + nPostMin + 1, length(days));

for iday = 1:length(days)
    % choice pref = 2*(auc-.5)
    choicePref_exc_aligned{iday} = -choicePref_all_alld_exc{iday}(eventI_stimOn_all(iday) - nPreMin  :  eventI_stimOn_all(iday) + nPostMin, :); % now ipsi is positive bc we do 1 minus, in the original model contra is positive
    choicePref_inh_aligned{iday} = -choicePref_all_alld_inh{iday}(eventI_stimOn_all(iday) - nPreMin  :  eventI_stimOn_all(iday) + nPostMin, :);
    
    % use area under ROC curve
    choicePref_exc_aligned{iday} = (0.5 + choicePref_exc_aligned{iday}/2); % now ipsi is positive bc we do - above
    choicePref_inh_aligned{iday} = (0.5 + choicePref_inh_aligned{iday}/2);
end


% This part is for roc hists
if trialHistAnalysis==0 && strcmp(mouse, 'fni16')
    cprintf('blue', 'Using variable ep to compute average ROC for setting histograms!\n')
    % For fni16 (that you used variable epochs for training SVM) compute ROC
    % average in ep of each day instead of during a fixed window
    % Average of ROC values during choiceTime-300:choiceTime for each neuron in each day
    exc_ep = [];
    inh_ep = [];
    for iday = 1:length(days)
        epnow = eventI_stimOn_all(iday) + ceil(ep_all(iday,:)/frameLength);

        exc_ep = [exc_ep, mean(choicePref_exc_aligned{iday}(epnow(1):epnow(2), :), 1)];
        inh_ep = [inh_ep, mean(choicePref_inh_aligned{iday}(epnow(1):epnow(2), :), 1)];
    end
end




%% Average across days

aveexc = cellfun(@(x)mean(x,2), choicePref_exc_aligned, 'uniformoutput',0); % average of neurons across each day
aveexc = cell2mat(aveexc);

aveinh = cellfun(@(x)mean(x,2), choicePref_inh_aligned, 'uniformoutput',0); % average of neurons across each day
aveinh = cell2mat(aveinh);

[h,p] = ttest2(aveexc',aveinh');
hh0 = h;
hh0(h==0) = nan;

% abs
%{
aveexc = cellfun(@(x)mean(abs(x),2), choicePref_exc_aligned, 'uniformoutput',0); % average of neurons across each day
aveexc = cell2mat(aveexc);

aveinh = cellfun(@(x)mean(abs(x),2), choicePref_inh_aligned, 'uniformoutput',0); % average of neurons across each day
aveinh = cell2mat(aveinh);
%}

%% Pool all days

excall = cell2mat(choicePref_exc_aligned);
inhall = cell2mat(choicePref_inh_aligned);
size(excall), size(inhall)

h = ttest2(excall', inhall');
hh = h;
hh(h==0) = nan;


%% Plots

figure('position',[10   556   792   383]);
subplot(121); hold on
h1 = boundedline(time_aligned, mean(aveexc,2), std(aveexc,0,2)/sqrt(size(aveexc,2)), 'b', 'alpha');
h2 = boundedline(time_aligned, mean(aveinh,2), std(aveinh,0,2)/sqrt(size(aveinh,2)), 'r', 'alpha');
a = get(gca, 'ylim');
plot(time_aligned, hh0*(a(2)-.05*diff(a)), 'k')
legend([h1,h2], 'Excitatory', 'Inhibitory')
xlabel('Time since stim onset (ms)')
ylabel('ROC performance')
title('mean+se days')

%%%
subplot(122); hold on
h1 = boundedline(time_aligned, mean(excall,2), std(excall,0,2)/sqrt(size(excall,2)), 'b', 'alpha');
h2 = boundedline(time_aligned, mean(inhall,2), std(inhall,0,2)/sqrt(size(inhall,2)), 'r', 'alpha');
a = get(gca, 'ylim');
plot(time_aligned, hh*(a(2)-.05*diff(a)), 'k')
legend([h1,h2], 'Excitatory', 'Inhibitory')
xlabel('Time since stim onset (ms)')
ylabel('ROC performance')
title('mean+se all neurons of all days')


if savefigs
    cd(fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse))
    if trialHistAnalysis==0
        savefig('curr_ave_ROC.fig')
    else
        savefig('prev_ave_ROC.fig')        
    end
end



%% Set vars for ROC histogram: average of ROC performance in the window [800 1100]ms for each neuron (to compare it with SVM performance in this same window)

eventI = nPreMin+1;
if trialHistAnalysis
%%%%%%%%%    NOTE: i think u should compute it for each day separtely ... instead of computing it on the aligned trace... this gives u more power... a lot of difference is expected to be at the beginning of the trace (not right before stim) and this is what you really do in svm.
    ep = 1: nPreMin+1; % all frames before the eventI
end

% use below to address the NOTE above.
%{
if trialHistAnalysis       
    % average in ep of each day instead of during a fixed window
    % Average of ROC values during choiceTime-300:choiceTime for each neuron in each day
    exc_ep = [];
    inh_ep = [];
    for iday = 1:length(days)
        epnow = 1 : eventI_stimOn_all(iday);

        a = 0.5 + -choicePref_all_alld_exc{iday}/2;
        b = 0.5 + -choicePref_all_alld_inh{iday}/2;
        
        exc_ep = [exc_ep, mean(a(epnow(1):epnow(2), :), 1)];
        inh_ep = [inh_ep, mean(b(epnow(1):epnow(2), :), 1)];
    end
end
%}

if trialHistAnalysis==0 && strcmp(mouse, 'fni17')
    cprintf('blue', 'using window: %d-%d to compute ROC distributions\n', ep_ms)
    epStartRel2Event = ceil(ep_ms(1)/frameLength); % the start point of the epoch relative to alignedEvent for training SVM. (500ms)
    epEndRel2Event = ceil(ep_ms(2)/frameLength); % the end point of the epoch relative to alignedEvent for training SVM. (700ms)
    ep = eventI+epStartRel2Event : eventI+epEndRel2Event; % frames on stimAl.traces that will be used for trainning SVM.    
end

if strcmp(mouse, 'fni17') || trialHistAnalysis==1
    cprintf('blue','Training epoch, rel2 stimOnset, is %.2f to %.2f ms\n', round((ep(1)-eventI)*frameLength), round((ep(end)-eventI)*frameLength))

    choicePref_exc_aligned_aveEP = cellfun(@(x)mean(x(ep,:),1), choicePref_exc_aligned, 'uniformoutput', 0); % average of ROC AUC during ep for each neuron
    choicePref_inh_aligned_aveEP = cellfun(@(x)mean(x(ep,:),1), choicePref_inh_aligned, 'uniformoutput', 0); % average of ROC AUC during ep for each neuron
    % a = mean(choicePref_exc_aligned{iday}(ep,:),1); % mean of each neuron during ep in day iday.

    %%%
    exc_ep = cell2mat(choicePref_exc_aligned_aveEP);
    inh_ep = cell2mat(choicePref_inh_aligned_aveEP);
end


cprintf('blue','Num exc neurons = %d, inh = %d\n', length(exc_ep), length(inh_ep))

[~, p] = ttest2(exc_ep, inh_ep);
cprintf('m','ROC at ep, pval= %.3f\n', p)

% fraction of neurons that carry >=10% info relative to chance (ie >=60% or <=40%)
fract_exc_inh_above10PercInfo = [mean(abs(exc_ep-.5)>.1) , mean(abs(inh_ep-.5)>.1)]


%% Histogram of ROC AUC for all neurons (averaged during ep)

bins = 0:.01:1;
[nexc, e] = histcounts(exc_ep, bins);
[ninh, e] = histcounts(inh_ep, bins);

x = mode(diff(bins))/2 + bins; x = x(1:end-1);
ye = nexc/sum(nexc);
yi = ninh/sum(ninh);
ye = smooth(ye);
yi = smooth(yi);

figure;
subplot(211), hold on
plot(x, ye)
plot(x, yi)
xlabel('ROC AUC')
ylabel('Fraction neurons')
legend('exc','inh')
plot([.5 .5],[0 max([ye;yi])], 'k:')
a = gca;

subplot(212), hold on
plot(x, cumsum(ye))
plot(x, cumsum(yi))
xlabel('ROC AUC')
ylabel('Cum fraction neurons')
legend('exc','inh')
plot([.5 .5],[0 max([ye;yi])], 'k:')
a = [a, gca];

linkaxes(a, 'x')


if savefigs
    cd(fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse))
    if trialHistAnalysis==0
        savefig('curr_dist_ROC.fig')
    else
        savefig('prev_dist_ROC.fig')        
    end
end

