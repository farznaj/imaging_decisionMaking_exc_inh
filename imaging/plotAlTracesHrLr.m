%% Set vars for plotting (for popClassifier_setXYall)

% savefigs = 0;
% loadPostNameVars = 1; % set to 1 if this script is not called from imaging_prep_analysis 
% traceType = '_'; % whether traces are S, C, or df/f 
% traceType = '_C_';
% traceType = '_S_';
% traceType = '_dfof_';

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
    ep_ms = []; %[809, 1109]; %[425, 725] % optional, it will be set according to min choice time if not provided.% training epoch relative to stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during ep ms after stimulus onset. [1000, 1300]; %[700, 900]; % [500, 700];
    % outcome2ana will be used if trialHistAnalysis is 0. When it is 1, by default we are analyzing past correct trials. If you want to change that, set it in the matlab code.
    outcome2ana = 'corr'; % '', corr', 'incorr' % trials to use for SVM training (all, correct or incorrect trials)
    strength2ana = 'all'; % 'all', easy', 'medium', 'hard' % What stim strength to use for training?
    thStimStrength = 3; % 2; % threshold of stim strength for defining hard, medium and easy trials.
    th_stim_dur = 0; % for fni18 u had to set it to 0! for fni16, fni17 it was 800; % min stim duration to include a trial in timeStimOnset
end
trs4project = 'trained'; % 'trained', 'all', 'corr', 'incorr' % trials that will be used for projections and the class accuracy trace; if 'trained', same trials that were used for SVM training will be used. "corr" and "incorr" refer to current trial's outcome, so they don't mean much if trialHistAnalysis=1.

thAct = 5e-4; %5e-4; % 1e-5 % neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.
% thTrsWithSpike = 1; % 3 % remove neurons that are active in <thSpTr trials.

pnev2load = []; % which pnev file to load: indicates index of date-sorted files: use 0 for latest. Set [] to load the latest one.

frameLength = 1000/30.9; % sec.


%% Make plots for average of correct and incorrect trials.

o = {'corr', 'incorr'};

for io = 1:length(o)

    outcome2ana = o{io};
    
    % set vars
    popClassifier_setXYall
%     close % close that stupid figure
    
    
    %%
    %%%%%%%%%%%%%%%%%% Set the traces that will be used for plots
    
    %% Load event-aligned traces
    
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
    
    
    % Set traces for specific neuron types: inhibitory, excitatory or all neurons
    if neuronType~=2
        traces_al_1stSide = traces_al_1stSide(:, nt, :);
        traces_al_go = traces_al_go(:, nt, :);
        traces_al_rew = traces_al_rew(:, nt, :);
        traces_al_incorrResp = traces_al_incorrResp(:, nt, :);
        traces_al_init = traces_al_init(:, nt, :);
    end
    
    
    
    
    
    %% Set the final traces for plotting (Remove trs and neurons to be excluded!)
    
    % trs4project = 'incorr' % 'trained', 'all', 'corr', 'incorr'
    
    % Data = scio.loadmat(postName, variable_names=['outcomes', 'allResp_HR_LR'])
    % choiceVecAll = (Data.pop('allResp_HR_LR').astype('float'))[0,:]
    
    % Set trials that will be used for projection traces
    
    if strcmp(trs4project, 'all')
        Xt = traces_al_stim;
        Xt_choiceAl = traces_al_1stSide;
        Xt_goAl = traces_al_go;
        Xt_rewAl = traces_al_rew;
        Xt_incorrRespAl = traces_al_incorrResp;
        Xt_initAl = traces_al_init;
        Xt_stimAl_all = traces_al_stimAll;
        choiceVecNow = choiceVecAll;
    elseif strcmp(trs4project, 'trained')
        Xt = traces_al_stim(:, :, ~trsExcluded);
        Xt_choiceAl = traces_al_1stSide(:, :, ~trsExcluded);
        Xt_goAl = traces_al_go(:, :, ~trsExcluded);
        Xt_rewAl = traces_al_rew(:, :, ~trsExcluded);
        Xt_incorrRespAl = traces_al_incorrResp(:, :, ~trsExcluded);
        Xt_initAl = traces_al_init(:, :, ~trsExcluded);
        Xt_stimAl_all = traces_al_stimAll(:, :, ~trsExcluded);
        choiceVecNow = Y;
    elseif strcmp(trs4project, 'corr')
        Xt = traces_al_stim(:, :, outcomes==1);
        Xt_choiceAl = traces_al_1stSide(:, :, outcomes==1);
        Xt_goAl = traces_al_go(:, :, outcomes==1);
        Xt_rewAl = traces_al_rew(:, :, outcomes==1);
        Xt_incorrRespAl = traces_al_incorrResp(:, :, outcomes==1);
        Xt_initAl = traces_al_init(:, :, outcomes==1);
        Xt_stimAl_all = traces_al_stimAll(:, :, outcomes==1);
        choiceVecNow = choiceVecAll(outcomes==1);
        
    elseif strcmp(trs4project, 'incorr')
        Xt = traces_al_stim(:, :, outcomes==0);
        Xt_choiceAl = traces_al_1stSide(:, :, outcomes==0);
        Xt_goAl = traces_al_go(:, :, outcomes==0);
        Xt_rewAl = traces_al_rew(:, :, outcomes==0);
        Xt_incorrRespAl = traces_al_incorrResp(:, :, outcomes==0);
        Xt_initAl = traces_al_init(:, :, outcomes==0);
        Xt_stimAl_all = traces_al_stimAll(:, :, outcomes==0);
        choiceVecNow = choiceVecAll(outcomes==0);
    end
    % Xt = traces_al_stim(:, :, np.sum(np.sum(np.isnan(traces_al_stim), axis =0), axis =0)==0);
    % Xt_choiceAl = traces_al_1stSide(:, :, np.sum(np.sum(np.isnan(traces_al_1stSide), axis =0), axis =0)==0);
    
    
    
    % Exclude non-active neurons (ie neurons that don't fire in any of the trials during ep)
    Xt = Xt(:,~NsExcluded,:);
    Xt_choiceAl = Xt_choiceAl(:,~NsExcluded,:);
    Xt_goAl = Xt_goAl(:,~NsExcluded,:);
    Xt_rewAl = Xt_rewAl(:,~NsExcluded,:);
    Xt_incorrRespAl = Xt_incorrRespAl(:,~NsExcluded,:);
    Xt_initAl = Xt_initAl(:,~NsExcluded,:);
    Xt_stimAl_all = Xt_stimAl_all(:,~NsExcluded,:);
    
    % Only include the randomly selected set of neurons
    %{
Xt = Xt(:,NsRand,:);
Xt_choiceAl = Xt_choiceAl(:,NsRand,:);
Xt_goAl = Xt_goAl(:,NsRand,:);
Xt_rewAl = Xt_rewAl(:,NsRand,:);
Xt_incorrRespAl = Xt_incorrRespAl(:,NsRand,:);
Xt_initAl = Xt_initAl(:,NsRand,:);
Xt_stimAl_all = Xt_stimAl_all(:,NsRand,:);
    %}
    
    
    % Divide data into high-rate (modeled as 1) and low-rate (modeled as 0) trials
    hr_trs = (choiceVecNow==1);
    lr_trs = (choiceVecNow==0);
    % print 'Projection traces have %d high-rate trials, and %d low-rate trials' %(np.sum(hr_trs), np.sum(lr_trs))
    
    
    
    % window of training (ep)
    win = (ep-eventI)*frameLength;
    
    
    
    
    %% Plot raw averages of population activity
    
    if doPlots
        % window of training (ep)
        % win = (ep-eventI)*frameLength
        
        % init-aligned projections and raw average
        figure('name', sprintf('Only %s; %d neurons; %d trials', outcome2ana, numNeurons, numTrials))
        subplot(3,2,1)
        a1 = squeeze(nanmean(Xt_initAl(:, :, hr_trs),  2)); % frames x trials
        tr1 = nanmean(a1,  2);
        tr1_se = nanstd(a1,  [], 2) / sqrt(numTrials);
        a0 = squeeze(nanmean(Xt_initAl(:, :, lr_trs),  2)); % frames x trials
        tr0 = nanmean(a0,  2);
        tr0_se = nanstd(a0,  [], 2) / sqrt(numTrials);
        h1=boundedline(time_aligned_init, tr1, tr1_se, 'b', 'alpha');
        h2=boundedline(time_aligned_init, tr0, tr0_se, 'r', 'alpha');
        xlabel('time aligned to init tone (ms)')
        legend([h1,h2], 'high rate', 'low rate'); set(legend, 'position', [0.4226    0.8873    0.1893    0.1190])
        title('Init tone')
        xlim([time_aligned_init(1), time_aligned_init(end)])
        yl = [min([tr1-tr1_se; tr0-tr0_se]), max([tr1+tr1_se; tr0+tr0_se])];
        plot([0, 0],[yl(1),yl(2)],'k:')
        ylim(yl)        
        
        
        % stim-aligned projections and raw average
        
        subplot(3,2,2) % I think you should use Xtsa here to make it compatible with the plot above.
        a1 = squeeze(nanmean(Xt(:, :, hr_trs),  2)); % frames x trials
        tr1 = nanmean(a1,  2);
        tr1_se = nanstd(a1,  [], 2) / sqrt(numTrials);
        a0 = squeeze(nanmean(Xt(:, :, lr_trs),  2)); % frames x trials
        tr0 = nanmean(a0,  2);
        tr0_se = nanstd(a0,  [], 2) / sqrt(numTrials);
        boundedline(time_aligned_stim, tr1, tr1_se, 'b', 'alpha')
        boundedline(time_aligned_stim, tr0, tr0_se, 'r', 'alpha')
        xlabel('time aligned to stim (ms)')
        title('Stim')
        xlim([time_aligned_stim(1), time_aligned_stim(end)])
        yl = [min([tr1-tr1_se; tr0-tr0_se]), max([tr1+tr1_se; tr0+tr0_se])];
        plot([0, 0],[yl(1),yl(2)],'k:')
        ylim(yl)
        
        
        % goTone-aligned projections and raw average
        
        subplot(3,2,3)
        a1 = squeeze(nanmean(Xt_goAl(:, :, hr_trs),  2)); % frames x trials
        tr1 = nanmean(a1,  2);
        tr1_se = nanstd(a1,  [], 2) / sqrt(numTrials);
        a0 = squeeze(nanmean(Xt_goAl(:, :, lr_trs),  2)); % frames x trials
        tr0 = nanmean(a0,  2);
        tr0_se = nanstd(a0,  [], 2) / sqrt(numTrials);
        boundedline(time_aligned_go, tr1, tr1_se, 'b', 'alpha')
        boundedline(time_aligned_go, tr0, tr0_se, 'r', 'alpha')
        xlabel('time aligned to go tone (ms)')
        title('Go tone')
        xlim([time_aligned_go(1), time_aligned_go(end)])
        yl = [min([tr1-tr1_se; tr0-tr0_se]), max([tr1+tr1_se; tr0+tr0_se])];
        plot([0, 0],[yl(1),yl(2)],'k:')
        ylim(yl)
        
       
        % choice-aligned projections and raw average
        
        subplot(3,2,4)
        a1 = squeeze(nanmean(Xt_choiceAl(:, :, hr_trs),  2)); % frames x trials
        tr1 = nanmean(a1,  2);
        tr1_se = nanstd(a1,  [], 2) / sqrt(numTrials);
        a0 = squeeze(nanmean(Xt_choiceAl(:, :, lr_trs),  2)); % frames x trials
        tr0 = nanmean(a0,  2);
        tr0_se = nanstd(a0,  [], 2) / sqrt(numTrials);
        boundedline(time_aligned_1stSide, tr1, tr1_se, 'b', 'alpha')
        boundedline(time_aligned_1stSide, tr0, tr0_se, 'r', 'alpha')
        xlabel('time aligned to choice (ms)')
        title('Choice')
        xlim([time_aligned_1stSide(1), time_aligned_1stSide(end)])
        yl = [min([tr1-tr1_se; tr0-tr0_se]), max([tr1+tr1_se; tr0+tr0_se])];
        plot([0, 0],[yl(1),yl(2)],'k:')
        ylim(yl)        
        
       
        % reward-aligned projections and raw average
       if strcmp(outcome2ana, 'corr')
            subplot(3,2,5)
            a1 = squeeze(nanmean(Xt_rewAl(:, :, hr_trs),  2)); % frames x trials
            tr1 = nanmean(a1,  2);
            tr1_se = nanstd(a1,  [], 2) / sqrt(numTrials);
            a0 = squeeze(nanmean(Xt_rewAl(:, :, lr_trs),  2)); % frames x trials
            tr0 = nanmean(a0,  2);
            tr0_se = nanstd(a0,  [], 2) / sqrt(numTrials);
            boundedline(time_aligned_rew, tr1, tr1_se, 'b', 'alpha')
            boundedline(time_aligned_rew, tr0, tr0_se, 'r', 'alpha')
            xlabel('time aligned to reward (ms)')
            title('Reward')
            xlim([time_aligned_rew(1), time_aligned_rew(end)])
            yl = [min([tr1-tr1_se; tr0-tr0_se]), max([tr1+tr1_se; tr0+tr0_se])];
            plot([0, 0],[yl(1),yl(2)],'k:')
            ylim(yl)        
        end
        
        % incommitResp-aligned projections and raw average
        if strcmp(outcome2ana, 'incorr')
            subplot(3,2,6)
            a1 = squeeze(nanmean(Xt_incorrRespAl(:, :, hr_trs),  2)); % frames x trials
            tr1 = nanmean(a1,  2);
            tr1_se = nanstd(a1,  [], 2) / sqrt(numTrials);
            a0 = squeeze(nanmean(Xt_incorrRespAl(:, :, lr_trs),  2)); % frames x trials
            tr0 = nanmean(a0,  2);
            tr0_se = nanstd(a0,  [], 2) / sqrt(numTrials);
            boundedline(time_aligned_incorrResp, tr1, tr1_se, 'b', 'alpha')
            boundedline(time_aligned_incorrResp, tr0, tr0_se, 'r', 'alpha')
            xlabel('time aligned to incorrResp (ms)')
            title('Incorr resp')
            xlim([time_aligned_incorrResp(1), time_aligned_incorrResp(end)])
            yl = [min([tr1-tr1_se; tr0-tr0_se]), max([tr1+tr1_se; tr0+tr0_se])];
            plot([0, 0],[yl(1),yl(2)],'k:')
            ylim(yl)
        end
        
        
        if savefigs
            savefig(fullfile(pd, 'figs',sprintf('caTraces%s%s_alHrLr', traceType, outcome2ana)))
        end
    end      
    
end


