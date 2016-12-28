% you need the following vars.
%{
trialHistAnalysis = 0;
iTiFlg = 2; % Only needed if trialHistAnalysis=1; short ITI, 1: long ITI, 2: all ITIs.
rmvInactiveNsDurEp = 0; % if 0 you wont remove any neurons; otherwise inactive neurons during ep will be removed. Set to 1 for svm analysis. otherwise set to 0.
setNsExcluded = 1; % if 1, NsExcluded will be set even if it is already saved.
% numSamples = 100; % number of iterations for finding the best c (inverse of regularization parameter)
neuronType = 2; % 0: excitatory, 1: inhibitory, 2: all types.    
% saveResults = 0; % save results in mat file.


doPlots = 1; % Whether to make plots or not.

if trialHistAnalysis==1 % more parameters are specified in popClassifier_trialHistory.m
%        iTiFlg = 1; % 0: short ITI, 1: long ITI, 2: all ITIs.
    epEnd_rel2stimon_fr = 0; % 3; % -2 % epEnd = eventI + epEnd_rel2stimon_fr
else
    % not needed to set ep_ms here, later you define it as [choiceTime-300 choiceTime]ms % we also go 30ms back to make sure we are not right on the choice time!
    ep_ms = [809, 1109]; %[425, 725] % optional, it will be set according to min choice time if not provided.% training epoch relative to stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during ep ms after stimulus onset. [1000, 1300]; %[700, 900]; % [500, 700]; 
    % outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.
    outcome2ana = 'corr'; % '', corr', 'incorr' % trials to use for SVM training (all, correct or incorrect trials)
    strength2ana = 'all'; % 'all', easy', 'medium', 'hard' % What stim strength to use for training?
    thStimStrength = 3; % 2; % threshold of stim strength for defining hard, medium and easy trials.
    th_stim_dur = 800; % min stim duration to include a trial in timeStimOnset
end
trs4project = 'trained'; % 'trained', 'all', 'corr', 'incorr' % trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1. 

thAct = 5e-4; %5e-4; % 1e-5 % neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.
% thTrsWithSpike = 1; % 3 % remove neurons that are active in <thSpTr trials.

pnev2load = []; % which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.
   
frameLength = 1000/30.9; % sec.
%}

%% Set the names of imaging-related .mat file names.
% remember the last saved pnev mat file will be the pnevFileName

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
% pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
disp(pnev_n)
cd(fileparts(imfilename))

moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));

[pd, pnev_n] = fileparts(pnevFileName);
postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));



%% Load matlab variables: event-aligned traces, inhibitRois, outcomes,  choice, etc
%     - traces are set in set_aligned_traces.m matlab script.
% Load time of some trial events
load(postName, 'timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'time1stSideTry')


% Load stim-aligned_allTrials traces, frames, frame of event of interest
if trialHistAnalysis==0
    load(postName, 'stimAl_noEarlyDec')
    stimAl_allTrs = stimAl_noEarlyDec;
else
    load(postName, 'stimAl_allTrs')
end
eventI = stimAl_allTrs.eventI;
traces_al_stimAll = stimAl_allTrs.traces;
time_aligned_stim = stimAl_allTrs.time;    % '(frames x units x trials)'
fprintf('size of stimulus-aligned traces: %d %d %d (frames x units x trials)\n', size(traces_al_stimAll))

traces_al_stim = traces_al_stimAll;


% Load outcomes and choice (allResp_HR_LR) for the current trial
load(postName, 'outcomes', 'allResp_HR_LR')
choiceVecAll = allResp_HR_LR;  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.
fprintf('Current outcome: %d correct choices; %d incorrect choices\n', sum(outcomes==1), sum(outcomes==0))

if trialHistAnalysis
    % Load trialHistory structure to get choice vector of the previous trial
    load(postName, 'trialHistory')
    choiceVec0All = trialHistory.choiceVec0;
end


% Set trials strength and identify trials with stim strength of interest
if trialHistAnalysis==0
    load(postName, 'stimrate', 'cb')
    
    s = stimrate-cb; % how far is the stimulus rate from the category boundary?
    if strcmp(strength2ana, 'easy')
        str2ana = (abs(s) >= (max(abs(s)) - thStimStrength));
    elseif strcmp(strength2ana, 'hard')
        str2ana = (abs(s) <= thStimStrength);
    elseif strcmp(strength2ana, 'medium')
        str2ana = ((abs(s) > thStimStrength) & (abs(s) < (max(abs(s)) - thStimStrength)));
    else
        str2ana = ones(1, length(outcomes));
        
        fprintf('Number of trials with stim strength of interest = %i\n', sum(str2ana))
        %     fprintf('Stim rates for training = {}'.format(unique(stimrate[str2ana]))
    end
end





%% Set the time window for training SVM (ep) and traces_al_stim

if trialHistAnalysis==1
    % either of the two below (stimulus-aligned and initTone-aligned) would be fine
    % eventI = DataI['initToneAl'].eventI - 1
    eventI = stimAl_allTrs.eventI; % remember to subtract 1! matlab vs python indexing!
    epEnd = eventI + epEnd_rel2stimon_fr; %- 2 % to be safe for decoder training for trial-history analysis we go upto the frame before the stim onset
    % epEnd = DataI['initToneAl'].eventI - 2 % to be safe for decoder training for trial-history analysis we go upto the frame before the initTone onset
    ep = 1:epEnd;
%     fprint('training epoch is {} ms'.format(round((ep-eventI)*frameLength))
    
    ep_ms = round((ep-eventI)*frameLength); % so it is the same format as ep_ms when trialHistAnalysis is 0
    
else
    % Set ep_ms if it is not provided: [choiceTime-300 choiceTime]ms % we also go 30ms back to make sure we are not right on the choice time!
    % by doing this you wont need to set ii below.
    
    % We first set to nan timeStimOnset of trials that anyway wont matter bc their outcome is  not of interest. we do this to make sure these trials dont affect our estimate of ep_ms
    if strcmp(outcome2ana, 'corr')
        cprintf('blue', 'Analyzing only correct trials.\n')
        timeStimOnset(outcomes~=1) = nan; % analyze only correct trials.
    elseif strcmp(outcome2ana, 'incorr')
        cprintf('blue', 'Analyzing only incorrect trials.\n')
        timeStimOnset(outcomes~=0) = nan; % analyze only incorrect trials.
    else
        cprintf('blue', 'Analyzing both correct and incorrect trials.\n')        
    end
    
    if ~exist('ep_ms', 'var')
        ep_ms = [floor(nanmin(time1stSideTry-timeStimOnset))-30-300, floor(nanmin(time1stSideTry-timeStimOnset))-30];
        fprintf('Training window: [%d %d] ms\n', ep_ms(1), ep_ms(2))
    end
    epStartRel2Event = ceil(ep_ms(1)/frameLength); % the start point of the epoch relative to alignedEvent for training SVM. (500ms)
    epEndRel2Event = ceil(ep_ms(2)/frameLength); % the end point of the epoch relative to alignedEvent for training SVM. (700ms)
    ep = eventI+epStartRel2Event : eventI+epEndRel2Event; % frames on stimAl.traces that will be used for trainning SVM.
    fprintf('training epoch, rel2 stimOnset, is %.2f to %.2f ms\n', round((ep(1)-eventI)*frameLength) - frameLength/2, round((ep(end)-eventI)*frameLength) - frameLength/2)
end



%% Exclude some trials from traces_al_stim (choice earlier than ep(end) and stimDur shorter than th_stim_dur).

if trialHistAnalysis==0    
     % This criteria makes sense if you want to be conservative; otherwise if ep=[1000 1300]ms, go tone will definitely be before ep end, and you cannot have the following criteria.
    % Make sure in none of the trials Go-tone happened before the end of training window (ep)
    i = (timeCommitCL_CR_Gotone - timeStimOnset) <= ep_ms(end);
    %{
    if sum(i)>0:
        fprintf('Excluding %i trials from timeStimOnset bc their goTone is earlier than ep end' %(sum(i))
    %     timeStimOnset[i] = nan;  % by setting to nan, the aligned-traces of these trials will be computed as nan.
    else:
        print('No trials with go tone before the end of ep. Good :)')
    %}
    
    % Make sure in none of the trials choice (1st side try) happened before the end of training window (ep)
    ii = (time1stSideTry - timeStimOnset) <= ep_ms(end);
    if sum(ii)>0
        fprintf('Excluding %i trials from timeStimOnset bc their choice is earlier than ep end\n', sum(ii))
        %     timeStimOnset[i] = nan;  % by setting to nan, the aligned-traces of these trials will be computed as nan.
    else
        fprintf('No trials with choice before the end of ep. Good :)\n')
    end
    
    % Make sure trials that you use for SVM (decoding upcoming choice from
    % neural responses during stimulus) have a certain stimulus duration. Of
    % course stimulus at least needs to continue until the end of ep.
    % go with either 900 or 800ms. Since the preference is to have at least
    % ~100ms after ep which contains stimulus and without any go tones, go with 800ms
    % bc in many sessions go tone happened early... so you will loose lots of
    % trials if you go with 900ms.
    % th_stim_dur = 800; % min stim duration to include a trial in timeStimOnset
    
    if doPlots
        load(postName, 'timeReward', 'timeCommitIncorrResp')
        
        figure; hold on
%         subplot(1,2,1); 
        plot(timeCommitCL_CR_Gotone - timeStimOnset)
        plot(timeStimOffset - timeStimOnset, 'r')
        plot(time1stSideTry - timeStimOnset, 'm')
        plot(timeReward - timeStimOnset)
        plot(timeCommitIncorrResp - timeStimOnset)
        plot([1, length(timeCommitCL_CR_Gotone)],[th_stim_dur, th_stim_dur], 'g:')
        plot([1, length(timeCommitCL_CR_Gotone)],[ep_ms(end), ep_ms(end)], 'k:')
        xlabel('Trial')
        ylabel('Time relative to stim onset (ms)')
        legend('goTone', 'stimOffset', '1stSideTry', 'reward', 'incorr', 'th\_stim\_dur', 'epoch end')
        % minStimDurNoGoTone = nanmin(timeCommitCL_CR_Gotone - timeStimOnset); % this is the duration after stim onset during which no go tone occurred for any of the trials.
        % fprintf('minStimDurNoGoTone = %.2f ms' %minStimDurNoGoTone
    end
    
    % Exclude trials whose stim duration was < th_stim_dur
    j = (timeStimOffset - timeStimOnset) < th_stim_dur;
    if sum(j)>0
        fprintf('Excluding %i trials from timeStimOnset bc their stimDur < %dms\n', sum(j), th_stim_dur)
        %     timeStimOnset[j] = nan;
    else
        fprintf('No trials with stimDur < %dms. Good :)\n', th_stim_dur)
    end
    
    
    % Set trials to be removed from traces_al_stimAll
    % toRmv = (i+j+ii)~=0;
    toRmv = (j+ii)~=0; fprintf('Not excluding %i trials whose goTone is earlier than ep end\n', sum(i))
    fprintf('Final: %i trials excluded in traces_al_stim\n', sum(toRmv))
    
    
    % Set traces_al_stim for SVM classification of current choice.
    traces_al_stim(:,:,toRmv) = nan;
    %     traces_al_stim[:,:,outcomes==-1] = nan
    % print(shape(traces_al_stim))
    
    
    %{
    % Set ep
    if len(ep_ms)==0: % load ep from matlab
        % Load stimulus-aligned traces, frames, frame of event of interest, and epoch over which we will average the responses to do SVM analysis
        Data = scio.loadmat(postName, variable_names=['stimAl'],squeeze_me=True,struct_as_record=False)
        % eventI = Data['stimAl'].eventI - 1 % remember difference indexing in matlab and python!
        % traces_al_stim = Data['stimAl'].traces.astype('float') % traces_al_stim
        % time_aligned_stim = Data['stimAl'].time.astype('float')

        ep = Data['stimAl'].ep - 1
        ep_ms = round((ep-eventI)*frameLength).astype(int)

    else: % set ep here:
    %}
end    

 
%% Load inhibitRois and set traces for specific neuron types: inhibitory, excitatory or all neurons

load(moreName, 'inhibitRois')
% fprintf('%d inhibitory, %d excitatory; %d unsure class' %(sum(inhibitRois==1), sum(inhibitRois==0), sum(isnan(inhibitRois)))

    
% Set traces for specific neuron types: inhibitory, excitatory or all neurons
if neuronType~=2
    nt = (inhibitRois==neuronType); % 0: excitatory, 1: inhibitory, 2: all types.
    % good_excit = inhibitRois==0;
    % good_inhibit = inhibitRois==1;        
    
    traces_al_stim = traces_al_stim(:, nt, :);
    traces_al_stimAll = traces_al_stimAll(:, nt, :);
else
    nt = 1:size(traces_al_stim,2);
end


%% Set X (trials x neurons) and Y (trials x 1) for training the SVM classifier.
%     X matrix (size trials x neurons) that contains neural responses at different trials.
%     Y choice of high rate (modeled as 1) and low rate (modeled as 0)

%% Set choiceVec0  (Y: the response vector)

if trialHistAnalysis
    choiceVec0 = choiceVec0All(:,iTiFlg); % choice on the previous trial for short (or long or all) ITIs
    choiceVec0S = choiceVec0All(:,1);
    choiceVec0L = choiceVec0All(:,2);
else % set choice for the current trial
    choiceVec0 = allResp_HR_LR';  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    % choiceVec0 = transpose(allResp_HR_LR);  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    
    if strcmp(outcome2ana, 'corr')
        choiceVec0(outcomes~=1) = nan; % analyze only correct trials.
    elseif strcmp(outcome2ana, 'incorr')
        choiceVec0(outcomes~=0) = nan; % analyze only incorrect trials.   
    end
    choiceVec0(~str2ana) = nan;   
    % Y = choiceVec0
    % print(choiceVec0.shape)
end


%% Set spikeAveEp0  (X: the predictor matrix (trials x neurons) that shows average of spikes for a particular epoch for each trial and neuron.)

if trialHistAnalysis
    % either of the two cases below should be fine (init-aligned traces or stim-aligned traces.)
    spikeAveEp0 = squeeze(nanmean(traces_al_stimAll(ep,:,:), 1))'; % trials x neurons    
    % spikeAveEp0 = squeeze(nanmean(traces_al_init(ep,:,:), 1))'; % trials x neurons    
else
    spikeAveEp0 = squeeze(nanmean(traces_al_stim(ep,:,:), 1))'; % trials x neurons    
end
% X = spikeAveEp0;
fprintf('Size of spikeAveEp0 (trs x neurons): %d x %d \n', size(spikeAveEp0))



%% Set trsExcluded and exclude them to set X and Y; trsExcluded are trials that are nan either in traces or in choice vector.

% Identify nan trials
trsExcluded = (sum(isnan(spikeAveEp0), 2) + isnan(choiceVec0)) ~= 0; % NaN trials % trsExcluded
% fprintf(sum(trsExcluded), 'NaN trials'


%% Exclude nan trials

X = spikeAveEp0(~trsExcluded,:); % trials x neurons
Y = choiceVec0(~trsExcluded);
fprintf('%d high-rate choices, and %d low-rate choices\n', sum(Y==1), sum(Y==0))




%% Set NsExcluded : Identify neurons that did not fire in any of the trials (during ep) and then exclude them. Otherwise they cause problem for feature normalization.
% thAct and thTrsWithSpike are parameters that you can play with.

if ~rmvInactiveNsDurEp
    NsExcluded = false(1, size(traces_al_stim,2)); % use this if you don't want to exclude any neurons.
    fprintf('chose not to remove any neurons!\n')
else
    % If it is already saved, load it (the idea is to use the same NsExcluded for all the analyses of a session). Otherwise set it.
    if trialHistAnalysis==0
        svmnowname = ['svmCurrChoice_allN', '_*-', pnevFileName(end-31:end)];
    else
        svmnowname = ['svmPrevChoice_allN_allITIs', '_*-', pnevFileName(end-31:end)];
    end
    svmName = dir(fullfile(pd, 'svm', svmnowname));
    [~,i] = sort([svmName.datenum], 'descend');
    svmName = svmName(i(1)).name; % so the latest file is the 1st one.



    if setNsExcluded==0 && ~isempty(svmName) % NsExcluded is already set and saved % 0: %    
    %     svmName = svmName(1);  % get the latest file
        fprintf('loading NsExcluded from file %s\n', svmName)
        load(svmName, 'NsExcluded')
        NsExcluded = NsExcluded(nt);

        stdX = std(X, [], 1); % define stdX for all neurons; later we reset it only including active neurons
        if min(stdX(~NsExcluded)) < thAct % make sure the loaded NsExcluded makes sense; ie stdX of ~NsExcluded is above thAct
            error('min of stdX= %.8f; not supposed to be <%d (thAct)!', min(stdX), thAct)
        end

    else    
        fprintf('NsExcluded not saved, so setting it here\n')

        if trialHistAnalysis && iTiFlg~=2
            % set X for short-ITI and long-ITI cases (XS,XL).
            trsExcludedS = (sum(isnan(spikeAveEp0), 2) + isnan(choiceVec0S)) ~= 0; 
            XS = spikeAveEp0(~trsExcludedS,:); % trials x neurons
            trsExcludedL = (sum(isnan(spikeAveEp0), 2) + isnan(choiceVec0L)) ~= 0; 
            XL = spikeAveEp0(~trsExcludedL,:); % trials x neurons

            % Define NsExcluded as neurons with low stdX for either short ITI or long ITI trials. 
            % This is to make sure short and long ITI cases will include the same set of neurons.
            stdXS = std(XS, [], 1);
            stdXL = std(XL, [], 1);

            NsExcluded = sum([stdXS < thAct; stdXL < thAct], 1)~=0; % if a neurons is non active for either short ITI or long ITI trials, exclude it.

        else

            % Define NsExcluded as neurons with low stdX
            stdX = std(X, [], 1);
            NsExcluded = stdX < thAct;
            % sum(stdX < thAct)

            %{
            % Set nonActiveNs, ie neurons whose average activity during ep is less than thAct.
        %     spikeAveEpAveTrs = nanmean(spikeAveEp0, 1); % 1 x units % response of each neuron averaged across epoch ep and trials.
            spikeAveEpAveTrs = nanmean(X, 1); % 1 x units % response of each neuron averaged across epoch ep and trials.
            % thAct = 5e-4; % 1e-5 %quantile(spikeAveEpAveTrs, .1);
            nonActiveNs = spikeAveEpAveTrs < thAct;
            fprintf('\t%d neurons with ave activity in ep < %.5f' %(sum(nonActiveNs), thAct)
            sum(nonActiveNs)

            % Set NsFewTrActiv, ie neurons that are active in very few trials (by active I mean average activity during epoch ep)
            % thTrsWithSpike = 1; % 3; % ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.
            nTrsWithSpike = sum(X > thAct, 1) % 0 % shows for each neuron, in how many trials the activity was above 0.
            NsFewTrActiv = (nTrsWithSpike < thTrsWithSpike) % identify neurons that were active fewer than thTrsWithSpike.
            fprintf('\t%d neurons are active in < %i trials' %(sum(NsFewTrActiv), thTrsWithSpike)

            % Now set the final NxExcluded: (neurons to exclude)
            NsExcluded = (NsFewTrActiv + nonActiveNs)~=0
            %}
        end
    end
end


fprintf('%d = Final % non-active neurons\n', sum(NsExcluded))
% a = size(spikeAveEp0,2) - sum(NsExcluded);
fprintf('Using %d out of %d neurons; Fraction excluded = %.2f\n', length(spikeAveEp0)-sum(NsExcluded), length(spikeAveEp0), sum(NsExcluded)/length(spikeAveEp0))


fprintf('%i, %i, %i: #original inh, excit, unsure\n', sum(inhibitRois==1), sum(inhibitRois==0), sum(isnan(inhibitRois)))
% Check what fraction of inhibitRois are excluded, compare with excitatory neurons.
if neuronType==2    
    fprintf('%i, %i, %i: #excluded inh, excit, unsure\n', sum(inhibitRois(NsExcluded)==1), sum(inhibitRois(NsExcluded)==0), sum(isnan(inhibitRois(NsExcluded))))
    fprintf('%.2f, %.2f, %.2f: fraction excluded inh, excit, unsure\n', sum(inhibitRois(NsExcluded)==1)/sum(inhibitRois==1), sum(inhibitRois(NsExcluded)==0)/sum(inhibitRois==0), sum(isnan(inhibitRois(NsExcluded)))/sum(isnan(inhibitRois)))
end



%% Exclude non-active neurons from X and set inhRois (ie neurons that don't fire in any of the trials during ep)

X = X(:,~NsExcluded);
disp(size(X));
    
% Set inhRois which is same as inhibitRois but with non-active neurons excluded. (it has same size as X)
if neuronType==2
    inhRois = inhibitRois(~NsExcluded);
    % fprintf('Number: inhibit = %d, excit = %d, unsure = %d' %(sum(inhRois==1), sum(inhRois==0), sum(isnan(inhRois)))
    % fprintf('Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f' %(fractInh, fractExc, fractUn)    
end
    

numTrials = sum(~trsExcluded);
numNeurons = sum(~NsExcluded);
fprintf('%d neurons; %d trials\n', numNeurons, numTrials)


















