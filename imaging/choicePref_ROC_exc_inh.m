trialHistAnalysis = 0;
nt = 0;
thStimStrength = 0; % 2; % what stim strength you want to use for computing choice pref.

frameLength = 1000/30.9; % sec.
makeplots = 0;
useEqualNumTrs = 0; % if true, equal number of trials for HR and LR will be used to compute ROC.

mouse = 'fni17';
days = {'151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', ...
    '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', ...
    '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'};


%%
if nt==0
    choicePref_all_alld_exc = cell(1, length(days));
elseif nt==1
    choicePref_all_alld_inh = cell(1, length(days));
end
eventI_stimOn_all = nan(1,length(days));


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



%%
% save roc_curr_stimstr2 eventI_stimOn_all choicePref_all_alld_exc choicePref_all_alld_inh


%%
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

figure; hold on
h1 = boundedline(time_aligned, mean(aveexc,2), std(aveexc,0,2)/sqrt(size(aveexc,2)), 'g', 'alpha');
h2 = boundedline(time_aligned, mean(aveinh,2), std(aveinh,0,2)/sqrt(size(aveinh,2)), 'r', 'alpha');
a = get(gca, 'ylim');
plot(time_aligned, hh0*(a(2)-.05*diff(a)), 'k')
legend([h1,h2], 'Excitatory', 'Inhibitory')
xlabel('Time since stim onset (ms)')
ylabel('ROC performance')

%%
figure; hold on
h1 = boundedline(time_aligned, mean(excall,2), std(excall,0,2)/sqrt(size(excall,2)), 'g', 'alpha');
h2 = boundedline(time_aligned, mean(inhall,2), std(inhall,0,2)/sqrt(size(inhall,2)), 'r', 'alpha');
a = get(gca, 'ylim');
plot(time_aligned, hh*(a(2)-.05*diff(a)), 'k')
legend([h1,h2], 'Excitatory', 'Inhibitory')
xlabel('Time since stim onset (ms)')
ylabel('ROC performance')


