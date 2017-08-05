function [ispec, itimed, sspec, stimed, gspec, gtimed, cspec, ctimed, rspec, rtimed, pspec, ptimed, ni1,ni0,ns1,ns0,ng1,ng0,nc1,nc0,nr1,nr0,np1,np0] = ...
    selHrLr(mouse, imagingFolder, mdfFileNumber, regressBins)
% Compute response selectivity for HR vs LR trials for the trial averaged responses for each neuron.
% Selectivity = (hr-lr)/(hr+lr)

%{
mouse = 'fni17';
imagingFolder = '150820';
mdfFileNumber = [1];
%}

io = 1; % analyze correct (1) or incorrect trials (2)?
% th = 0; % min number of trials for each hr and and lr in order to compute selectivity.

if ~exist('regressBins', 'var')
    regressBins = 3;
end


%%
loadPostNameVars = 1; % set to 1 if this script is not called from imaging_prep_analysis
trialHistAnalysis = 0;
o = {'corr', 'incorr'};
outcome2ana = o{io};
% strength2ana = 'all'; % 'all', easy', 'medium', 'hard' % What stim strength to use for training?
% neuronType = 2; % 0: excitatory, 1: inhibitory, 2: all types.
% savefigs = 0;


%%
%{
signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
% [pd, pnev_n] = fileparts(pnevFileName);
[md,date_major] = fileparts(imfilename);
cd(md)
% r = repmat('%03d-', 1, length(mdfFileNumber)); r(end) = []; date_major = sprintf(['%s_', r], imagingFolder, mdfFileNumber);
[~, pnev_n] = fileparts(pnevFileName);
postName = fullfile(md, sprintf('post_%s.mat', pnev_n));
%}


%% Set the names of imaging-related .mat file names.
% remember the last saved pnev mat file will be the pnevFileName

if loadPostNameVars
    signalCh = 2; % because you get A from channel 2, I think this should be always 2.
    pnev2load = [];
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    [pd, pnev_n] = fileparts(pnevFileName);
    disp(pnev_n)
    cd(fileparts(imfilename))
    
    %     moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
    
    [pd, pnev_n] = fileparts(pnevFileName);
    postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
end


%% Load matlab variables: event-aligned traces, inhibitRois, outcomes,  choice, etc
%     - traces are set in set_aligned_traces.m matlab script. % Load time of some trial events

% Load outcomes and choice (allResp_HR_LR) for the current trial
if loadPostNameVars
    load(postName, 'outcomes', 'allResp_HR_LR')
end
%{
outcomes = outcomes(1:length(all_data_sess{1}));
allResp_HR_LR = allResp_HR_LR(1:length(all_data_sess{1}));
%}
choiceVecAll = allResp_HR_LR;  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.
cprintf('blue', 'Current outcome: %d correct choices; %d incorrect choices\n', sum(outcomes==1), sum(outcomes==0))
cprintf('blue','\tCorr: %d LR; %d HR\n',  sum(choiceVecAll(outcomes==1)==0), sum(choiceVecAll(outcomes==1)==1))
cprintf('blue','\tIncorr: %d LR; %d HR\n',  sum(choiceVecAll(outcomes==0)==0), sum(choiceVecAll(outcomes==0)==1))


%% Load event-aligned traces

% if loadPostNameVars
%     load(postName, 'timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'time1stSideTry')

if loadPostNameVars
    if trialHistAnalysis==0
        load(postName, 'stimAl_noEarlyDec') % Load stim-aligned_allTrials traces, frames, frame of event of interest
        stimAl_allTrs = stimAl_noEarlyDec;
    else
        load(postName, 'stimAl_allTrs')
    end
else
    stimAl_allTrs = stimAl_noEarlyDec;
end
eventI = stimAl_allTrs.eventI;
% traces_al_stimAll = stimAl_allTrs.traces;
traces_al_stim = stimAl_allTrs.traces;
time_aligned_stim = stimAl_allTrs.time;    % '(frames x units x trials)'
% fprintf('size of stimulus-aligned traces: %d %d %d (frames x units x trials)\n', size(traces_al_stimAll))
% traces_al_stim = traces_al_stimAll;


% Load 1stSideTry-aligned traces, frames, frame of event of interest
% use firstSideTryAl_COM to look at changes-of-mind (mouse made a side lick without committing it)
if loadPostNameVars
    load(postName, 'firstSideTryAl')
end
traces_al_1stSide = firstSideTryAl.traces;
time_aligned_1stSide = firstSideTryAl.time;
% print(shape(traces_al_1stSide))


% Load goTone-aligned traces, frames, frame of event of interest
% use goToneAl_noStimAft to make sure there was no stim after go tone.
if loadPostNameVars
    load(postName, 'goToneAl')
end
traces_al_go = goToneAl.traces;
time_aligned_go = goToneAl.time;
% print(shape(traces_al_go))


% Load reward-aligned traces, frames, frame of event of interest
if loadPostNameVars
    load(postName, 'rewardAl')
end
traces_al_rew = rewardAl.traces;
time_aligned_rew = rewardAl.time;
% print(shape(traces_al_rew))


% Load commitIncorrect-aligned traces, frames, frame of event of interest
if loadPostNameVars
    load(postName, 'commitIncorrAl')
end
traces_al_incorrResp = commitIncorrAl.traces;
time_aligned_incorrResp = commitIncorrAl.time;
% print(shape(traces_al_incorrResp))


% Load initiationTone-aligned traces, frames, frame of event of interest
if loadPostNameVars
    load(postName, 'initToneAl')
end
traces_al_init = initToneAl.traces;
time_aligned_init = initToneAl.time;
% print(shape(traces_al_init))
% DataI = Data
%{
if trialHistAnalysis:
    % either of the two below (stimulus-aligned and initTone-aligned) would be fine
    % eventI = DataI['initToneAl'].eventI
    eventI = DataS['stimAl_allTrs'].eventI
    epEnd = eventI + epEnd_rel2stimon_fr %- 2 % to be safe for decoder training for trial-history analysis we go upto the frame before the stim onset
    % epEnd = DataI['initToneAl'].eventI - 2 % to be safe for decoder training for trial-history analysis we go upto the frame before the initTone onset
    ep = arange(epEnd+1)
    fprintf('training epoch is {} ms'.format(round((ep-eventI)*frameLength))
%}


%% Analyze correct or incorrect trials?

if strcmp(outcome2ana, 'corr')
    Xt = traces_al_stim(:, :, outcomes==1);
    Xt_choiceAl = traces_al_1stSide(:, :, outcomes==1);
    Xt_goAl = traces_al_go(:, :, outcomes==1);
    Xt_rewAl = traces_al_rew(:, :, outcomes==1);
    Xt_incorrRespAl = traces_al_incorrResp(:, :, outcomes==1);
    Xt_initAl = traces_al_init(:, :, outcomes==1);
    %     Xt_stimAl_all = traces_al_stimAll(:, :, outcomes==1);
    choiceVecNow = choiceVecAll(outcomes==1);
    
elseif strcmp(outcome2ana, 'incorr')
    Xt = traces_al_stim(:, :, outcomes==0);
    Xt_choiceAl = traces_al_1stSide(:, :, outcomes==0);
    Xt_goAl = traces_al_go(:, :, outcomes==0);
    Xt_rewAl = traces_al_rew(:, :, outcomes==0);
    Xt_incorrRespAl = traces_al_incorrResp(:, :, outcomes==0);
    Xt_initAl = traces_al_init(:, :, outcomes==0);
    %     Xt_stimAl_all = traces_al_stimAll(:, :, outcomes==0);
    choiceVecNow = choiceVecAll(outcomes==0);
end
% Xt = traces_al_stim(:, :, np.sum(np.sum(np.isnan(traces_al_stim), axis =0), axis =0)==0);
% Xt_choiceAl = traces_al_1stSide(:, :, np.sum(np.sum(np.isnan(traces_al_1stSide), axis =0), axis =0)==0);

% Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
%{
Xt = Xt(:,~NsExcluded,:);
Xt_choiceAl = Xt_choiceAl(:,~NsExcluded,:);
Xt_goAl = Xt_goAl(:,~NsExcluded,:);
Xt_rewAl = Xt_rewAl(:,~NsExcluded,:);
Xt_incorrRespAl = Xt_incorrRespAl(:,~NsExcluded,:);
Xt_initAl = Xt_initAl(:,~NsExcluded,:);
Xt_stimAl_all = Xt_stimAl_all(:,~NsExcluded,:);
%}


%%
% numTrials = sum(~trsExcluded);
% numNeurons = sum(~NsExcluded);
numTrials = size(Xt,3);
numNeurons = size(Xt,2);
cprintf('blue', '%d neurons; %d trials\n', numNeurons, numTrials)


% Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
hr_trs = (choiceVecNow==1);
lr_trs = (choiceVecNow==0);

%%% Compute number of valid trials for each each aligned trace
ni1 = sum(~sum(isnan(squeeze(nanmean(Xt_initAl(:,:,hr_trs), 1))), 1));
ni0 = sum(~sum(isnan(squeeze(nanmean(Xt_initAl(:,:,lr_trs), 1))), 1));
fprintf('init #trs: hr=%d, lr=%d\n', ni1, ni0)

ns1 = sum(~sum(isnan(squeeze(nanmean(Xt(:,:,hr_trs), 1))), 1));
ns0 = sum(~sum(isnan(squeeze(nanmean(Xt(:,:,lr_trs), 1))), 1));
fprintf('stim #trs: hr=%d, lr=%d\n', ns1, ns0)

ng1 = sum(~sum(isnan(squeeze(nanmean(Xt_goAl(:,:,hr_trs), 1))), 1));
ng0 = sum(~sum(isnan(squeeze(nanmean(Xt_goAl(:,:,lr_trs), 1))), 1));
fprintf('go #trs: hr=%d, lr=%d\n', ng1, ng0)

% since change-of-mind trials are excluded from time1stSideTry, hr and lr
% trials for choice_aligned might be a subset of all other alignments. so
% we need to set hr and lr trials for choice_aligned separately.
f1 = find(hr_trs);
f0 = find(lr_trs);
v1 = ~sum(isnan(squeeze(nanmean(Xt_choiceAl(:,:,hr_trs), 1))), 1); % non-nan trials in hr_trs. NaN trials are change-of-mind trials.
v0 = ~sum(isnan(squeeze(nanmean(Xt_choiceAl(:,:,lr_trs), 1))), 1);
hr_trs_c = false(size(hr_trs));  % length of all correct (or incorrect) trials.
lr_trs_c = false(size(hr_trs)); 
hr_trs_c(f1(v1)) = true; % only valid hr_trs are set to 1.
lr_trs_c(f0(v0)) = true; % only valid lr_trs are set to 1.
nc1 = sum(v1);
nc0 = sum(v0);
fprintf('choice #trs: hr=%d, lr=%d\n', nc1, nc0)

nr1 = sum(~sum(isnan(squeeze(nanmean(Xt_rewAl(:,:,hr_trs), 1))), 1));
nr0 = sum(~sum(isnan(squeeze(nanmean(Xt_rewAl(:,:,lr_trs), 1))), 1));
fprintf('reward #trs: hr=%d, lr=%d\n', nr1, nr0)

np1 = sum(~sum(isnan(squeeze(nanmean(Xt_incorrRespAl(:,:,hr_trs), 1))), 1));
np0 = sum(~sum(isnan(squeeze(nanmean(Xt_incorrRespAl(:,:,lr_trs), 1))), 1));
fprintf('incorrResp #trs: hr=%d, lr=%d\n', np1, np0)


%% Downsample traces, average across each trial category, and compute selectivity to HR vs LR.

hr = find(hr_trs);
lr = find(lr_trs);
nresamp = 1e2; % number of bootstrap samples

fprintf('init...\n')
[ispec, itimed] = setSpec_bs(Xt_initAl, time_aligned_init, choiceVecNow, regressBins, hr, lr, nresamp, ni1, ni0);
fprintf('stim...\n')
[sspec, stimed] = setSpec_bs(Xt, time_aligned_stim, choiceVecNow, regressBins, hr, lr, nresamp, ns1, ns0);
fprintf('go tone...\n')
[gspec, gtimed] = setSpec_bs(Xt_goAl, time_aligned_go, choiceVecNow, regressBins, hr, lr, nresamp, ng1, ng0);
fprintf('choice...\n')
[cspec, ctimed] = setSpec_bs(Xt_choiceAl, time_aligned_1stSide, choiceVecNow, regressBins, find(hr_trs_c), find(lr_trs_c), nresamp, nc1, nc0);
fprintf('reward...\n')
[rspec, rtimed] = setSpec_bs(Xt_rewAl, time_aligned_rew, choiceVecNow, regressBins, hr, lr, nresamp, nr1, nr0);
fprintf('incorr resp...\n')
[pspec, ptimed] = setSpec_bs(Xt_incorrRespAl, time_aligned_incorrResp, choiceVecNow, regressBins, hr, lr, nresamp, np1, np0);



%% Bootstrap version of setSpec

    function [ispec, itimed] = setSpec_bs(Xt_initAl, time_aligned_init, choiceVecNow, regressBins, hr, lr, nresamp, ni1, ni0)
        
        th = 10; % min number of trials for each hr and and lr in order to compute selectivity.
%         ispec = nan(floor(size(Xt_initAl,1)/regressBins), nresamp);
%         itimed = nan(size(ispec,1),1);        
        [ispec0, itimed] = setSpec(Xt_initAl, time_aligned_init, choiceVecNow, regressBins);
        ispec = nan(length(ispec0), nresamp);
        
        if (ni1 >= th && ni0 >= th)            
            ispec(:,1) = ispec0;            
            for isamp = 2:nresamp
                i1 = randi(ni1, 1, ni1); % randomly select (with replacement) ni1 trials.
                i0 = randi(ni0, 1, ni0);
                trs = [hr(i1), lr(i0)];

                x = Xt_initAl(:,:,trs);
                c = choiceVecNow(trs);

                [ispec0] = setSpec(x, time_aligned_init, c, regressBins);
%                 if any(isnan(ispec0)), error('why'), end
                ispec(:,isamp) = ispec0;
            end
        end
        
        %%
        function [spec, timed] = setSpec(X, ti, choiceVecNow, rb)
            
            % Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
            hr_trs = (choiceVecNow==1);
            lr_trs = (choiceVecNow==0);
            
            % Average activity across all HR and all LR trials.
            a1 = squeeze(nanmean(X(:, :, hr_trs),  2)); % frames x trials
            tr1 = nanmean(a1,  2);
            % tr1_se = nanstd(a1,  [], 2) / sqrt(numTrials);
            a0 = squeeze(nanmean(X(:, :, lr_trs),  2)); % frames x trials
            tr0 = nanmean(a0,  2);
            % tr0_se = nanstd(a0,  [], 2) / sqrt(numTrials);
            
            eventI = find(ti==0);
            
            
            %% Downsample (average in a few frames)
            
            % rb = 3; % regressBins
            % t = tr1(1: rb*floor(length(tr1)/rb));
            % a = reshape(t, rb, []);
            % t = time_aligned_stim(1: rb*floor(length(tr1)/rb));
            % b = reshape(t, rb, []);
            
            
            % NOTE: maybe it makes more sense to have first element of Post start with
            % frame 0, in case there is any response on frame 0!
            
            % tr1
            t = tr1(eventI - rb*floor(eventI/rb) +1  :  eventI); % last element: frame 0
            a = reshape(t, rb, []);
            t = tr1(eventI+1 : max(eventI+1 : rb : length(tr1))); % frame 1 to end
            a = [a, reshape(t(1:rb*floor(length(t)/rb)), rb, [])]; % column corresponding to frame 0, includes frames [-2,-1,0].
            tr1d = mean(a,1);
            
            % tr0
            t = tr0(eventI - rb*floor(eventI/rb) +1  :  eventI); % last element: frame 0
            a = reshape(t, rb, []);
            t = tr0(eventI+1 : max(eventI+1 : rb : length(tr1))); % frame 1 to end %      eventI+1 : rb*floor(length(tr1)/rb)
            a = [a, reshape(t(1:rb*floor(length(t)/rb)), rb, [])]; % column corresponding to frame 0, includes frames [-2,-1,0].
            tr0d = mean(a,1);
            
            % time
            t = ti(eventI - rb*floor(eventI/rb) +1  :  eventI); % last element: frame 0
            b = reshape(t, rb, []);
            t = ti(eventI+1 : max(eventI+1 : rb : length(tr1))); % frame 1 to end   % eventI+1 : rb*floor(length(tr1)/rb)
            b = [b, reshape(t(1:rb*floor(length(t)/rb)), rb, [])]; % column corresponding to frame 0, includes frames [-2,-1,0].
            timed = mean(b,1);
            
            
            %% Compute selectivity of response to HR vs LR trials (hr-lr)/(hr+lr)
            
            spec = (tr1d - tr0d) ./ (tr1d + tr0d); % how hr and lr resps are different.
            
            
        end               
    end
end

