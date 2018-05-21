% "choicePref_ROC_exc_inh.m" is already run and ROC vars are saved. 
% Here we load the mat files to set vars for plotting ROC results.

% Follow this script by the following two to make plots for each mouse and all mice:
% choicePref_ROC_exc_inh_plotsEachMouse
% choicePref_ROC_exc_inh_plotsAllMice


%% Load ROC vars and set vars required for plotting for each mouse %%%%%%

outcome2ana = 'corr'; %''; % 'corr'; 'incorr'; '';
% alFR = 'initAl'; % the firing rate traces were aligned on what

doChoicePref = 0; %2; 
    % doChoicePref=0;  % use area under ROC curve % cares about ipsi bigger than contra or vice versa;  ipsi is above 0.5
    % doChoicePref=1;  % use choice pref = 2(AUC-.5) % cares about ipsi bigger than contra or vice versa
    % doChoicePref=2;  % compute abs deviation of AUC from chance (.5); % doesnt care about ipsi bigger than contra or vice versa
fni18_rmvDay4 = 0; % if 1, remove 4th day of fni18.
eachMouse_do_savefigs = [1,0]; % Whether to make plots and to save them for each mouse
sumMice_do_savefigs = [1,0]; % Whether to make plots and to save them for the summary of mice
% normX = 1; % load FRs % after downsampling set max peak to 1. Makes sense to do, since traces before downsampling are normalized to max    

chAl = 1;


thMinTrs = 10; % days with fewer than this number wont go into analysis      
mice = {'fni16','fni17','fni18','fni19'};
thStimStrength = 0;
plotchance = 0; % it didnt make almost any difference so go with shuffled not chance % if 0, look at shuffled (tr shuffled, if imbalance hr and lr, it would be different from chance); % if 1, plot chance (where 0s and 1s were manually made the same number). % redefine _shfl vars as _chance vars, bc we want to look at chance values


%%
frameLength = 1000/30.9; % sec.
regressBins = round(100/frameLength); % 100ms # set to nan if you don't want to downsample.
% dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/ROC';
% dirn0fr = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/FR';
dm0 = char(strcat(join(mice, '_'),'_'));
% doplots = 1; % make plots for each mouse 
if doChoicePref==1  % use choice pref = 2*(auc-.5) % care about ipsi bigger than contra or vice versa.   % now ipsi is positive bc we do minus, in the original model contra is positive
    namc = 'choicePref';  yy = 0;
elseif doChoicePref==2  % Compute abs deviation of AUC from chance (.5); we want to compute |AUC-.5|, since choice pref = 2*(auc-.5), therefore |auc-.5| = 1/2 * |choice pref|
    namc = 'absDevAUC';   yy = [];
elseif doChoicePref==0  % use area under ROC curve  % choice pref = 2*(auc-.5), so  AUC = choice pref/2 + .5   % now ipsi is positive bc we do minus, in the original model contra is positive
    namc = 'AUC';   yy = .5;
elseif doChoicePref==0  % use area under ROC curve  % choice pref = 2*(auc-.5), so  AUC = choice pref/2 + .5   % now ipsi is positive bc we do minus, in the original model contra is positive
    namc = 'AUC';   yy = .5;
end

if strcmp(outcome2ana, '')
    namc = strcat(namc, '_allOutcome');
elseif strcmp(outcome2ana, 'corr')
    namc = strcat(namc, '_corr');
elseif strcmp(outcome2ana, 'incorr')
    namc = strcat(namc, '_incorr');
end
    
% cod = get(groot,'defaultAxesColorOrder'); 
% set(groot,'defaultAxesColorOrder',cod)
% default color order
cod = [0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% nowStr = datestr(now, 'yymmdd-HHMMSS');
imfni18 = find(strcmp(mice, 'fni18')); 
zScoreX = 0; % whether neural activity was z scored or not.
if zScoreX
    namz = '_zscoredX';
else
    namz = '';
end

if chAl
    al = 'chAl'; 
elseif initAl==1
    al = 'initAl'; 
end
        
% if normX, nmd = '_norm2max'; else, nmd = ''; end

if strcmp(outcome2ana, 'corr')
    o2a = '_corr'; 
elseif strcmp(outcome2ana, 'incorr')
    o2a = '_incorr';  
else
    o2a = '_allOutcome';
end      


%%
nPreMin_allMice = nan(1, length(mice));
time_aligned_allMice = cell(1, length(mice));

choicePref_exc_aligned_allMice = cell(1, length(mice));
choicePref_inh_aligned_allMice = cell(1, length(mice));
choicePref_exc_aligned_allMice_shfl0 = cell(1, length(mice));
choicePref_inh_aligned_allMice_shfl0 = cell(1, length(mice));
choicePref_exc_aligned_allMice_shfl = cell(1, length(mice));
choicePref_inh_aligned_allMice_shfl = cell(1, length(mice));

nowStr_allMice = cell(1, length(mice));
mnTrNum_allMice = cell(1, length(mice));
days_allMice = cell(1, length(mice));
corr_ipsi_contra_allMice = cell(1, length(mice));

exc_prob_dataROC_from_shflDist = cell(1, length(mice));
inh_prob_dataROC_from_shflDist = cell(1, length(mice));
%{
fr_exc_aligned_allMice = cell(1, length(mice));
fr_inh_aligned_allMice = cell(1, length(mice));

ipsiTrs_allDays_allMice = cell(1, length(mice));
contraTrs_allDays_allMice = cell(1, length(mice));
%}


%%
% im = 1;
for im = 1:length(mice)

    mouse = mice{im};
    [~,~,dirn] = setImagingAnalysisNames(mouse, 'analysis', []); 
    
    
    %% Load FR vars
    %{
    dirn = fullfile(dirn0fr, mouse);
    namv = sprintf('FR%s_curr_%s%s_stimstr%d%s_%s_*.mat', nmd, alFR,o2a,thStimStrength,namz,mouse);    
    a = dir(fullfile(dirn,namv));
    a = a(end); % use the latest saved file
    namatfr = a.name;
    
    load(fullfile(dirn, namatfr), 'X_svm_all_alld_exc', 'X_svm_all_alld_inh', 'ipsiTrs_allDays', 'contraTrs_allDays') % each cell is for a day and has size: frs x units x trials
    ipsiTrs_allDays_allMice{im} = ipsiTrs_allDays;
    contraTrs_allDays_allMice{im} = contraTrs_allDays;
    %}
    
    %% Set dir for loading ROC vars
    
%     dirn = fullfile(dirn0, mouse);
    namv = sprintf('ROC_curr_%s%s_stimstr%d%s_%s_*.mat', al,o2a,thStimStrength,namz,mouse);    
    a = dir(fullfile(dirn,namv));
    a = a(end); % use the latest saved file
    namatf = a.name;
    fprintf('%s\n',namatf)    
    
    %%% Set nowStr_allMice
    nam = namatf(end-4-12:end-4);
    if fni18_rmvDay4 && im==imfni18
        nam = strcat('fni18NoDay4_', nam);
    end
    if plotchance
        nam = strcat('chance_', nam);
    end
    if zScoreX
        nam = strcat('zScored_', nam);
    end
    nowStr_allMice{im} = nam;

    %%% Set days for each mouse
    if strcmp(mouse, 'fni16')
        days = {'150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'}; %'150914_1-2' : don't analyze!   
    elseif strcmp(mouse, 'fni17')
        days = {'150814_1', '150817_1', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};    
    elseif strcmp(mouse, 'fni18')
        days = {'151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'}; % alldays
    elseif strcmp(mouse, 'fni19')    
        days = {'150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'};    
    end
    days_allMice{im} = days;

    
    %% Load ROC vars for this mouse
    
    clear('corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'choicePref_all_alld_exc', 'choicePref_all_alld_inh', 'choicePref_all_alld_exc_shfl', 'choicePref_all_alld_inh_shfl', 'choicePref_all_alld_exc_chance', 'choicePref_all_alld_inh_chance')
    load(fullfile(dirn, namatf), 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'choicePref_all_alld_exc', 'choicePref_all_alld_inh', 'choicePref_all_alld_exc_shfl', 'choicePref_all_alld_inh_shfl', 'choicePref_all_alld_exc_chance', 'choicePref_all_alld_inh_chance')  
    
    if plotchance % redefine _shfl vars as _chance vars, bc we want to look at chance values
        choicePref_all_alld_exc_shfl = choicePref_all_alld_exc_chance;
        choicePref_all_alld_inh_shfl = choicePref_all_alld_inh_chance;
    end
    if isempty(choicePref_all_alld_inh_shfl{1})
        doshfl = 0;
    else
        doshfl = 1;
    end
    
    % get rid of the bad day of fni18
    if fni18_rmvDay4 && im==imfni18 %&& iday==4 % set this day to nan
        corr_ipsi_contra(4,:) = thMinTrs-1; % set it to a value lower than thMinTrs, so it gets removed!
    end
    
    nsamps = size(choicePref_all_alld_exc_shfl{1},3);        
    mnTrNum = min(corr_ipsi_contra,[],2); % min number of trials of the 2 class

    mnTrNum_allMice{im} = mnTrNum;
    corr_ipsi_contra_allMice{im} = corr_ipsi_contra;
        
    
    %% Prep to align all traces of all days (for each mouse) on common eventI

    % run the 1st section of this script
    % cd /home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC
    % load('roc_curr.mat')
    % load('roc_prev.mat')

    nPost = nan(1,length(days));
    for iday = 1:length(days)
        nPost(iday) = size(choicePref_all_alld_exc{iday},1) - eventI_ds_allDays(iday);
    end
    nPostMin = min(nPost);

    nPreMin = min(eventI_ds_allDays)-1;
    disp([nPreMin, nPostMin])


    %% Set downsampled time_aligned for the aligned trace for each mouse (assuming that choicePref was computed on downsampled traces).

    totLen = nPreMin + nPostMin +1;
    eventI = min(eventI_allDays);

    time_trace = frameLength * ((1 : ceil(regressBins*(totLen+1))) - eventI); % time_trace = time_aligned_1stSide

    % set frames before frame0 (not including it)
    e = eventI-1;
    f = e - regressBins*floor(e/regressBins) + 1 : e; % 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
    x = time_trace(f); % time_trace including frames before frame0
    T1 = length(x);
    tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames before frame0
    xdb = squeeze(mean(reshape(x, [regressBins, tt]), 1)); % downsampled time_trace inclusing frames before frame0


    % set frames after frame0 (including it)
    lenPost = length(time_trace) - (eventI-1);
%     lenPost = length(time_trace) - eventI;
    f = eventI : (eventI-1) + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins    
%     f = eventI+1 : eventI + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins    
    x = time_trace(f); % time_trace including frames after frame0
    T1 = length(x);
    tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames after frame0
    xda = squeeze(mean(reshape(x, [regressBins, tt]), 1)); % downsampled time_trace inclusing frames after frame0

    % set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frame0, and x at frames after frame0
    time_trace_d = cat(2, xdb, xda);
%     time_trace_d = cat(2, xdb, 0, xda);    
    time_aligned = time_trace_d(1:totLen);
    %{
    a = -frameLength * (0:nPreMin); a = a(end:-1:1);
    b = frameLength * (1:nPostMin);
    time_aligned = [a,b];
    %}
    size(time_aligned)
            

    %% Align FRs of all days on the common eventI; exclude days with too few trials.
    %%% REMEMBER: you are doing this weird thing for convenience: 
        % for days with few trials, we use nans for arbitrarily 3 neurons and 3 trials!
    %{        
    fr_exc_aligned = cell(1,length(days)); % each cell: nFrames_aligned x neurons x trials (for days with few trials, we use nans and arbitrarily 3 neurons!)
    fr_inh_aligned = cell(1,length(days));
    
    for iday = 1:length(days)
        if mnTrNum(iday,:) >= thMinTrs                        
            % data: frs x ns x trs
            fr_exc_aligned{iday} = X_svm_all_alld_exc{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :, :); 
            fr_inh_aligned{iday} = X_svm_all_alld_inh{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :, :); 
        else
            %%% REMEMBER: you are doing this weird thing for convenience: 
            % for days with few trials, we use nans and arbitrarily 3 neurons!
            fr_exc_aligned{iday} = nan(nPreMin + nPostMin + 1 , 3, 3); % set to nan so number of neurons and trials dont matter... I just picked 3.
            fr_inh_aligned{iday} = nan(nPreMin + nPostMin + 1 , 3, 3);
        end
    end
    %}
    
    
    %% Align choice pref traces of all days on the common eventI, exclude days with too few trials.
    %%% REMEMBER: you are doing this weird thing for convenience: 
    % for days with few trials, we use nans and arbitrarily 3 neurons!
    %%%% REMEMBER: ipsi will be positive and contra negative (because we
    %%%% add a negative sign here).    
    
    choicePref_exc_aligned = cell(1,length(days)); % each cell: nFrames_aligned x neurons (for days with few trials, we use nans and arbitrarily 3 neurons!)
    choicePref_inh_aligned = cell(1,length(days)); % nan(nPreMin + nPostMin + 1, length(days));
    if doshfl
        choicePref_exc_aligned_shfl0 = cell(1,length(days)); % each cell: nFrames_aligned x neurons x samps (for days with few trials, we use nans and arbitrarily 3 neurons!)
        choicePref_inh_aligned_shfl0 = cell(1,length(days));
        choicePref_exc_aligned_shfl = cell(1,length(days));
        choicePref_inh_aligned_shfl = cell(1,length(days));
    end
    for iday = 1:length(days)
        if mnTrNum(iday) >= thMinTrs            
            if doChoicePref==1  % use choice pref = 2*(auc-.5) % care about ipsi bigger than contra or vice versa
                % now ipsi is positive bc we do minus, in the original model contra is positive
%                 namc = 'choicePref';  yy = 0;
                
                % data: frs x ns
                choicePref_exc_aligned{iday} = -choicePref_all_alld_exc{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :); % now ipsi is positive bc we do 1 minus, in the original model contra is positive
                choicePref_inh_aligned{iday} = -choicePref_all_alld_inh{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :);
                
                if doshfl % shfl: frs x ns x samps
                    choicePref_exc_aligned_shfl0{iday} = -choicePref_all_alld_exc_shfl{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :,:); % now ipsi is positive bc we do 1 minus, in the original model contra is positive
                    choicePref_inh_aligned_shfl0{iday} = -choicePref_all_alld_inh_shfl{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :,:);                
                    % mean across samps: frs x ns
                    choicePref_exc_aligned_shfl{iday} = mean(choicePref_exc_aligned_shfl0{iday},3);
                    choicePref_inh_aligned_shfl{iday} = mean(choicePref_inh_aligned_shfl0{iday},3);
                end
                
            elseif doChoicePref==2  % Compute abs deviation of AUC from chance (.5); we want to compute
                % |AUC-.5|, since choice pref = 2*(auc-.5), therefore |auc-.5| = 1/2 * |choice pref|
%                 namc = 'absDevAUC';   yy = [];                
                
                % data: frs x ns
                choicePref_exc_aligned{iday} = .5*abs(-choicePref_all_alld_exc{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :)); % now ipsi is positive bc we do 1 minus, in the original model contra is positive
                choicePref_inh_aligned{iday} = .5*abs(-choicePref_all_alld_inh{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :));
                
                if doshfl
                    % shfl: frs x ns x samps: you must first take average
                    % across samples, then compute abs; you cannot first do
                    % abs for each sample, and then compute average across
                    % samples... this way the small differences from 0, wont be
                    % averaged out because you are first taking abs!!!
                    a = mean(choicePref_all_alld_exc_shfl{iday},3); % frames x neurons (averaged across samps)
                    a = a(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :,:); % alignedFrs x neurons
                    choicePref_exc_aligned_shfl{iday} = .5*abs(-a); % frs x ns  (now ipsi is positive bc we do 1 minus, in the original model contra is positive)

                    a = mean(choicePref_all_alld_inh_shfl{iday},3); % frames x neurons (averaged across samps)
                    a = a(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :,:); % alignedFrs x neurons
                    choicePref_inh_aligned_shfl{iday} = .5*abs(-a); % frs x ns  (now ipsi is positive bc we do 1 minus, in the original model contra is positive)

                    % the following is wrong, see the explanation above...
                    warning('check this: your note says: the following is wrong, see the explanation above')
                    choicePref_exc_aligned_shfl0{iday} = .5*abs(-choicePref_all_alld_exc_shfl{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :,:)); % now ipsi is positive bc we do 1 minus, in the original model contra is positive
                    choicePref_inh_aligned_shfl0{iday} = .5*abs(-choicePref_all_alld_inh_shfl{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :,:));                
                end
                
            elseif doChoicePref==0  % use area under ROC curve  % choice pref = 2*(auc-.5), so  AUC = choice pref/2 + .5 
                % now ipsi is above 0.5 bc we do minus, in the original model contra is positive
%                 namc = 'AUC';   yy = .5;
                
                % data: frs x ns
                a = -choicePref_all_alld_exc{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :);
                choicePref_exc_aligned{iday} = (0.5 + a/2); % now ipsi auc will be above 0.5 bc we do - above
                a = -choicePref_all_alld_inh{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :);
                choicePref_inh_aligned{iday} = (0.5 + a/2); % frames x neurons
                
                if doshfl % shfl: frs x ns x samps
                    
                    a = -choicePref_all_alld_exc_shfl{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :,:);
                    choicePref_exc_aligned_shfl0{iday} = (0.5 + a/2); % now ipsi auc will be above 0.5 bc we do - above
                    a = -choicePref_all_alld_inh_shfl{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :,:);
                    choicePref_inh_aligned_shfl0{iday} = (0.5 + a/2); % frames x neurons                
                    % mean across samps: frs x ns
                    choicePref_exc_aligned_shfl{iday} = mean(choicePref_exc_aligned_shfl0{iday},3);
                    choicePref_inh_aligned_shfl{iday} = mean(choicePref_inh_aligned_shfl0{iday},3);                

                    
                    %% Set probability of data ROC coming from the same distribution as shuffled (assuming shuffled is a normal distribution with mu and sigma equal to mean and std of shuffled ROC values for each neuron).
                    %%%%%%%% DONE FOR TIME -1 %%%%%%%%
                    
                    %%%% exc                    
                    shfl = squeeze(choicePref_exc_aligned_shfl0{iday}(nPreMin, :,:)); % neurons x samples % take ROC of shuffled and actual data at time -1
                    data = choicePref_exc_aligned{iday}(nPreMin, :); % 1 x neurons
                    [m, s] = normfit(shfl'); % 1 x neurons;  mean and std estimates of normal distribution parameters for each neuron
                    y = normpdf(data, m, s) ./  normpdf(m, m, s); % % 1 x neurons; for each neuron y is the probability of the normal distribution (defined by m and s) to have a value equal to data.                   
                    exc_prob_dataROC_from_shflDist{im}{iday} = y;
                    
                    %%%% inh                    
                    shfl = squeeze(choicePref_inh_aligned_shfl0{iday}(nPreMin, :,:)); % neurons x samples % take ROC of shuffled and actual data at time -1
                    data = choicePref_inh_aligned{iday}(nPreMin, :); % 1 x neurons
                    [m, s] = normfit(shfl'); % 1 x neurons;  mean and std estimates of normal distribution parameters for each neuron
                    y = normpdf(data, m, s) ./  normpdf(m, m, s); % % 1 x neurons; for each neuron y is the probability of the normal distribution (defined by m and s) to have a value equal to data.                   
                    inh_prob_dataROC_from_shflDist{im}{iday} = y;
                    

                end
            end
            
        else
            %%% REMEMBER: you are doing this weird thing for convenience: 
            % for days with few trials, we use nans and arbitrarily 3 neurons!
            choicePref_exc_aligned{iday} = nan(nPreMin + nPostMin + 1 , 3); % set to nan so number of neurons doesnt matter... I just picked 3.
            choicePref_inh_aligned{iday} = nan(nPreMin + nPostMin + 1 , 3);
            if doshfl
                choicePref_exc_aligned_shfl0{iday} = nan(nPreMin + nPostMin + 1 , 3, nsamps);
                choicePref_inh_aligned_shfl0{iday} = nan(nPreMin + nPostMin + 1 , 3, nsamps);
                choicePref_exc_aligned_shfl{iday} = nan(nPreMin + nPostMin + 1 , 3);
                choicePref_inh_aligned_shfl{iday} = nan(nPreMin + nPostMin + 1 , 3);            
            end
        end
    end



    %% Keep vars of all mice

    nPreMin_allMice(im) = nPreMin;
    time_aligned_allMice{im} = time_aligned;
    
    choicePref_exc_aligned_allMice{im} = choicePref_exc_aligned; % days; each day: frs x ns
    choicePref_inh_aligned_allMice{im} = choicePref_inh_aligned;
    if doshfl
        choicePref_exc_aligned_allMice_shfl0{im} = choicePref_exc_aligned_shfl0; % days; each day: frs x ns x samps
        choicePref_inh_aligned_allMice_shfl0{im} = choicePref_inh_aligned_shfl0;
        choicePref_exc_aligned_allMice_shfl{im} = choicePref_exc_aligned_shfl; % days; each day: frs x ns
        choicePref_inh_aligned_allMice_shfl{im} = choicePref_inh_aligned_shfl;    
    end
    %{
    fr_exc_aligned_allMice{im} = fr_exc_aligned;
    fr_inh_aligned_allMice{im} = fr_inh_aligned;
    %}
end





%% %% Set vars to make summary plots of all mice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set vars to align traces of all mice 

nPost = nan(1,length(mice));
for im = 1:length(mice)
    nPost(im) = length(time_aligned_allMice{im}) - (nPreMin_allMice(im)+1);
end
nPostMin = min(nPost);

nPreMin = min(nPreMin_allMice);
disp([nPreMin, nPostMin])
    
% set the time_aligned trace for all mice
time_al_b = time_aligned_allMice{im}(nPreMin_allMice(im)-nPreMin+1 : nPreMin_allMice(im));
time_al_a = time_aligned_allMice{im}(nPreMin_allMice(im)+1 : nPreMin_allMice(im)+1+nPostMin);

time_al = [time_al_b, time_al_a];


%% Align FRs
%{
fr_exc_al_allMice = cell(1, length(mice));
fr_inh_al_allMice = cell(1, length(mice));

for im = 1:length(mice)
    
    ev = nPreMin_allMice(im)+1; % eventI of the day-aligned traces for mouse im    
    for iday = 1:length(fr_exc_aligned_allMice{im})        
        fr_exc_al_allMice{im}{iday} = fr_exc_aligned_allMice{im}{iday}(ev - nPreMin  :  ev + nPostMin, :, :); % nAlignedFrs (across mice) x neurons
        fr_inh_al_allMice{im}{iday} = fr_inh_aligned_allMice{im}{iday}(ev - nPreMin  :  ev + nPostMin, :, :); % nAlignedFrs (across mice) x neurons
    end
end
%}

%% Align choice pref traces of all mice on the common eventI. (Traces of all days of each mouse are already aligned)
    
choicePref_exc_al_allMice = cell(1, length(mice));
choicePref_inh_al_allMice = cell(1, length(mice));
aveexc_allMice = cell(1, length(mice));
aveinh_allMice = cell(1, length(mice));
if doshfl
    choicePref_exc_al_allMice_shfl0 = cell(1, length(mice));
    choicePref_inh_al_allMice_shfl0 = cell(1, length(mice));
    choicePref_exc_al_allMice_shfl = cell(1, length(mice));
    choicePref_inh_al_allMice_shfl = cell(1, length(mice));
    % aveexc_allMice_shfl0 = cell(1, length(mice));
    % aveinh_allMice_shfl0 = cell(1, length(mice));
    aveexc_allMice_shfl = cell(1, length(mice));
    aveinh_allMice_shfl = cell(1, length(mice));
end

for im = 1:length(mice)
    
    ev = nPreMin_allMice(im)+1; % eventI of the day-aligned traces for mouse im
    
    for iday = 1:length(choicePref_exc_aligned_allMice{im})
        
        choicePref_exc_al_allMice{im}{iday} = choicePref_exc_aligned_allMice{im}{iday}(ev - nPreMin  :  ev + nPostMin, :); % nAlignedFrs (across mice) x neurons
        choicePref_inh_al_allMice{im}{iday} = choicePref_inh_aligned_allMice{im}{iday}(ev - nPreMin  :  ev + nPostMin, :);
        if doshfl % shfl
            choicePref_exc_al_allMice_shfl{im}{iday} = choicePref_exc_aligned_allMice_shfl{im}{iday}(ev - nPreMin  :  ev + nPostMin, :); % nAlignedFrs (across mice) x neurons
            choicePref_inh_al_allMice_shfl{im}{iday} = choicePref_inh_aligned_allMice_shfl{im}{iday}(ev - nPreMin  :  ev + nPostMin, :);        
            choicePref_exc_al_allMice_shfl0{im}{iday} = choicePref_exc_aligned_allMice_shfl0{im}{iday}(ev - nPreMin  :  ev + nPostMin, :,:); % nAlignedFrs (across mice) x neurons x samps
            choicePref_inh_al_allMice_shfl0{im}{iday} = choicePref_inh_aligned_allMice_shfl0{im}{iday}(ev - nPreMin  :  ev + nPostMin, :,:);
    %         %%%% average across samps for each neuron
    %         choicePref_exc_al_allMice_shfl{im}{iday} = mean(choicePref_exc_al_allMice_shfl0{im}{iday},3); % nAlignedFrs (across mice) x neurons
    %         choicePref_inh_al_allMice_shfl{im}{iday} = mean(choicePref_inh_al_allMice_shfl0{im}{iday},3);
        end
        
        %%%%%%%%%%% Average across neurons for each day and each frame
        aveexc_allMice{im}(:,iday) = mean(choicePref_exc_al_allMice{im}{iday},2); %nFrames x days 
        aveinh_allMice{im}(:,iday) = mean(choicePref_inh_al_allMice{im}{iday},2);       
%         seexc = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_exc_aligned, 'uniformoutput',0); % standard error across neurons
%         seexc = cell2mat(seexc); % frs x days  
        if doshfl  % shfl
            aveexc_allMice_shfl{im}(:,iday) = mean(choicePref_exc_al_allMice_shfl{im}{iday},2); %nFrames x days 
            aveinh_allMice_shfl{im}(:,iday) = mean(choicePref_inh_al_allMice_shfl{im}{iday},2);           
    %         aveexc_allMice_shfl0{im}{iday}(:,:) = squeeze(mean(choicePref_exc_al_allMice_shfl0{im}{iday},2)); %nDays; each days: nFrames x samps
    %         aveinh_allMice_shfl0{im}{iday}(:,:) = squeeze(mean(choicePref_inh_al_allMice_shfl0{im}{iday},2));       
    %         %%%% average across samps for each neuron
    %         aveexc_allMice_shfl{im}(:,iday) = mean(aveexc_allMice_shfl0{im}{iday},2); %nFrames x days
    %         aveinh_allMice_shfl{im}(:,iday) = mean(aveinh_allMice_shfl0{im}{iday},2);       
        end
    end    
end


%% Number of valid days for each mouse

numDaysAll = nan(1,length(mice));
numDaysGood = nan(1,length(mice));
for im = 1:length(mice)
    numDaysAll(im) = size(aveexc_allMice{im},2);
    numDaysGood(im) = sum(~isnan(aveexc_allMice{im}(1,:)));
end
numDaysAll
numDaysGood




%%
%%%%%%%%%%%%%% You will need the vars below for %%%%%%%%%%%%%%
%%%%%%%%%%%%%% choicePref_ROC_exc_inh_plotsAllMice %%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%%%%%%%%%% Work with average of neurons %%%%%%%%%%%%%%%%%%%%%%%
%% Average and se of traces across days for each mouse (traces are already averaged across neurons)

exc_avDays_eachMouse = cell2mat(cellfun(@(x)nanmean(x,2), aveexc_allMice, 'uniformoutput',0)); %nFrs x nMice
inh_avDays_eachMouse = cell2mat(cellfun(@(x)nanmean(x,2), aveinh_allMice, 'uniformoutput',0)); %nFrs x nMice

% ttest across mice to see if neuron-averaged,day-averaged roc traces are
% different between exc and inh 
[h,p0] = ttest2(exc_avDays_eachMouse', inh_avDays_eachMouse'); % 1 x nFrs
hh0_allm = h;
hh0_allm(h==0) = nan;
p0


%%% se across days for each mouse (traces are already averaged across neurons)
exc_sdDays_eachMouse = cell2mat(cellfun(@(x)nanstd(x,[],2), aveexc_allMice, 'uniformoutput',0)); %nFrs x nMice
exc_seDays_eachMouse = bsxfun(@rdivide, exc_sdDays_eachMouse, sqrt(numDaysGood));

inh_sdDays_eachMouse = cell2mat(cellfun(@(x)nanstd(x,[],2), aveinh_allMice, 'uniformoutput',0)); %nFrs x nMice
inh_seDays_eachMouse = bsxfun(@rdivide, inh_sdDays_eachMouse, sqrt(numDaysGood));

%%% Plot each mouse:
%{
for im=1:length(mice)
    figure
    hold on
    plot(time_al', exc_avDays_eachMouse(:,im), 'b')
    plot(time_al', inh_avDays_eachMouse(:,im), 'r')
end
%}

if doshfl %%%%%%%%% shfl % average across days (already averaged across samps)
    % reshape(x,size(x,1),[])
    exc_avDays_eachMouse_shfl = cell2mat(cellfun(@(x)nanmean(x,2), aveexc_allMice_shfl, 'uniformoutput',0)); %nFrs x nMice
    inh_avDays_eachMouse_shfl = cell2mat(cellfun(@(x)nanmean(x,2), aveinh_allMice_shfl, 'uniformoutput',0)); %nFrs x nMice

    % ttest across mice to see if neuron-averaged,day-averaged roc traces are
    % different between exc and inh 
    [h,p0s] = ttest2(exc_avDays_eachMouse_shfl', inh_avDays_eachMouse_shfl'); % 1 x nFrs
    hh0s_allm = h;
    hh0s_allm(h==0) = nan;
    p0s


    %%% se across days (already averaged across samps) for each mouse (traces are already averaged across neurons)
    exc_sdDays_eachMouse_shfl = cell2mat(cellfun(@(x)nanstd(x,[],2), aveexc_allMice_shfl, 'uniformoutput',0)); %nFrs x nMice
    exc_seDays_eachMouse_shfl = bsxfun(@rdivide, exc_sdDays_eachMouse_shfl, sqrt(numDaysGood));

    inh_sdDays_eachMouse_shfl = cell2mat(cellfun(@(x)nanstd(x,[],2), aveinh_allMice_shfl, 'uniformoutput',0)); %nFrs x nMice
    inh_seDays_eachMouse_shfl = bsxfun(@rdivide, inh_sdDays_eachMouse_shfl, sqrt(numDaysGood));
end



%% Pool neuron-averaged traces of all days of all mice

exc_allDaysPooled_allMice  = cell2mat(aveexc_allMice); %nFrs x nAllDays (of all mice)
inh_allDaysPooled_allMice = cell2mat(aveinh_allMice); %nFrs x nAllDays (of all mice)

% ttest across all days to see if neuron-averaged roc traces are different
% between exc and inh
[h,p] = ttest2(exc_allDaysPooled_allMice', inh_allDaysPooled_allMice'); % 1 x nFrs
hh_allm = h;
hh_allm(h==0) = nan;
% p


if doshfl %%% shfl
    exc_allDaysPooled_allMice_shfl  = cell2mat(aveexc_allMice_shfl); %nFrs x nAllDays (of all mice)
    inh_allDaysPooled_allMice_shfl = cell2mat(aveinh_allMice_shfl); %nFrs x nAllDays (of all mice)

    % ttest across all days to see if neuron-averaged roc traces are different
    % between exc and inh
    [h,ps] = ttest2(exc_allDaysPooled_allMice_shfl', inh_allDaysPooled_allMice_shfl'); % 1 x nFrs
    hhs_allm = h;
    hhs_allm(h==0) = nan;
    % ps
end



%%
%%%%%%%%%%%%%%%%%%%%%%% Work with individual neurons %%%%%%%%%%%%%%%%%%%%%%%
%% Pool all neurons of all days 

% for each mouse
exc_allNsDaysPooled_eachMouse = cellfun(@(x)cell2mat(x), choicePref_exc_al_allMice, 'uniformoutput',0); % cell: 1x4; each element: nFrs x nAllNs
inh_allNsDaysPooled_eachMouse = cellfun(@(x)cell2mat(x), choicePref_inh_al_allMice, 'uniformoutput',0);

% now pool all mice
exc_allNsDaysMicePooled = cell2mat(exc_allNsDaysPooled_eachMouse); % frs x allNs of allMice
inh_allNsDaysMicePooled = cell2mat(inh_allNsDaysPooled_eachMouse);


if doshfl %%% shfl (each neuron is already averaged across shuffles (samps))
    % for each mouse
    exc_allNsDaysPooled_eachMouse_shfl = cellfun(@(x)cell2mat(x), choicePref_exc_al_allMice_shfl, 'uniformoutput',0); % cell: 1x4; each element: nFrs x nAllNs
    inh_allNsDaysPooled_eachMouse_shfl = cellfun(@(x)cell2mat(x), choicePref_inh_al_allMice_shfl, 'uniformoutput',0);
    exc_allNsDaysPooled_eachMouse_shfl0 = cellfun(@(x)cell2mat(x), choicePref_exc_al_allMice_shfl0, 'uniformoutput',0); % cell: 1x4; each element: nFrs x nAllNs x nSamps
    inh_allNsDaysPooled_eachMouse_shfl0 = cellfun(@(x)cell2mat(x), choicePref_inh_al_allMice_shfl0, 'uniformoutput',0);
    
    % now pool all mice
    exc_allNsDaysMicePooled_shfl = cell2mat(exc_allNsDaysPooled_eachMouse_shfl); % nFrs x nAllNeurons
    inh_allNsDaysMicePooled_shfl = cell2mat(inh_allNsDaysPooled_eachMouse_shfl);
    exc_allNsDaysMicePooled_shfl0 = cell2mat(exc_allNsDaysPooled_eachMouse_shfl0); % nFrs x nAllNeurons x nSamples
    inh_allNsDaysMicePooled_shfl0 = cell2mat(inh_allNsDaysPooled_eachMouse_shfl0);    
end



%% Keep vars for outcome2ana : corr and incorr

if strcmp(outcome2ana, 'corr')
    choicePref_exc_al_allMice_corr = choicePref_exc_al_allMice;
    choicePref_inh_al_allMice_corr = choicePref_inh_al_allMice;
    time_al_corr = time_al;
    ipsi_contra_corr = corr_ipsi_contra_allMice;
    
elseif strcmp(outcome2ana, 'incorr')
    choicePref_exc_al_allMice_incorr = choicePref_exc_al_allMice;
    choicePref_inh_al_allMice_incorr = choicePref_inh_al_allMice;
    time_al_incorr = time_al;
    ipsi_contra_incorr = corr_ipsi_contra_allMice;    
end




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Probabilities of data ROC coming from the same distribution as shuffled %%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if doshfl
alpha = .05;

%% Each day: probability of data ROC coming from the same distribution as shuffled (assuming shuffled is a normal distribution with mu and sigma equal to mean and std of shuffled ROC values for each neuron).

if doChoicePref==0
    
    %%%% Set fractions of significantly tuned neurons 
    exc_fractSigTuned_eachDay = cell(1, length(mice));
    inh_fractSigTuned_eachDay = cell(1, length(mice));

    for im = 1:length(mice)    
        mnTrNum = mnTrNum_allMice{im};
        
        exc_fractSigTuned_eachDay{im} = nan(1, numDaysAll(im));
        inh_fractSigTuned_eachDay{im} = nan(1, numDaysAll(im));
    
        for iday = 1:numDaysAll(im)
            if mnTrNum(iday) >= thMinTrs
                exc_fractSigTuned_eachDay{im}(iday) = mean(exc_prob_dataROC_from_shflDist{im}{iday} <= alpha); % fraction of neurons that are significantly tuned (comparing AUC with shuffled AUCs)
                inh_fractSigTuned_eachDay{im}(iday) = mean(inh_prob_dataROC_from_shflDist{im}{iday} <= alpha);    
            end
        end
    end
end


%% Set probabilities of data ROC coming from the same distribution as shuffled for ipsi and contra preferring neurons separately
%%%% Plots are made in choicePref_ROC_exc_inh_plotsAllMice

if doChoicePref==0
    exc_ipsiPref_prob_dataFromShflDist_eachDay = cell(1, length(mice));
    exc_contraPref_prob_dataFromShflDist_eachDay = cell(1, length(mice));
    inh_ipsiPref_prob_dataFromShflDist_eachDay = cell(1, length(mice));
    inh_contraPref_prob_dataFromShflDist_eachDay = cell(1, length(mice));
    
    exc_fractSigTuned_ipsi_eachDay = cell(1, length(mice));
    exc_fractSigTuned_contra_eachDay = cell(1, length(mice));
    inh_fractSigTuned_ipsi_eachDay = cell(1, length(mice));
    inh_fractSigTuned_contra_eachDay = cell(1, length(mice));
    
    exc_fractSigIpsiTuned_eachDay = cell(1, length(mice));
    exc_fractSigContraTuned_eachDay = cell(1, length(mice));
    inh_fractSigIpsiTuned_eachDay = cell(1, length(mice));
    inh_fractSigContraTuned_eachDay = cell(1, length(mice));

    for im = 1:length(mice)
        mnTrNum = mnTrNum_allMice{im};

        exc_fractSigTuned_ipsi_eachDay{im} = nan(1, numDaysAll(im));
        exc_fractSigTuned_contra_eachDay{im} = nan(1, numDaysAll(im));
        inh_fractSigTuned_ipsi_eachDay{im} = nan(1, numDaysAll(im));
        inh_fractSigTuned_contra_eachDay{im} = nan(1, numDaysAll(im));

        exc_fractSigIpsiTuned_eachDay{im} = nan(1, numDaysAll(im));
        exc_fractSigContraTuned_eachDay{im} = nan(1, numDaysAll(im));
        inh_fractSigIpsiTuned_eachDay{im} = nan(1, numDaysAll(im));
        inh_fractSigContraTuned_eachDay{im} = nan(1, numDaysAll(im));
    
        for iday = 1:numDaysAll(im)
            if mnTrNum(iday) >= thMinTrs          

                % take AUC at time -1
                auc_tM1_exc = choicePref_exc_al_allMice{im}{iday}(nPreMin,:); % take AUC at time -1
                auc_tM1_inh = choicePref_inh_al_allMice{im}{iday}(nPreMin,:); % take AUC at time -1
                
                %%%%%%%%%%%%% Out of sig choice-tuned neurons, what fraction are ipsi and what fraction are contra
                %%% exc
                aa = (exc_prob_dataROC_from_shflDist{im}{iday} <= alpha);
                exc_fractSigIpsiTuned_eachDay{im}(iday) = mean(auc_tM1_exc(aa==1)>.5);
                exc_fractSigContraTuned_eachDay{im}(iday) = mean(auc_tM1_exc(aa==1)<.5);

                %%% inh
                aa = (inh_prob_dataROC_from_shflDist{im}{iday} <= alpha);
                inh_fractSigIpsiTuned_eachDay{im}(iday) = mean(auc_tM1_inh(aa==1)>.5);
                inh_fractSigContraTuned_eachDay{im}(iday) = mean(auc_tM1_inh(aa==1)<.5);    

                
                %%%%%%%%%%%%% Fraction of Ns with AUC >.5 (or <0.5) that are significantly tuned (comparing AUC with shuffled AUCs)
                %%% exc                
                exc_ipsiPref_prob_dataFromShflDist_eachDay{im}{iday} = exc_prob_dataROC_from_shflDist{im}{iday}(auc_tM1_exc > .5); % get prob of significancy for ipsi-preferring neurons (ie neurons whose AUC is > .5 but are not necessarily significantly tuned!)
                exc_contraPref_prob_dataFromShflDist_eachDay{im}{iday} = exc_prob_dataROC_from_shflDist{im}{iday}(auc_tM1_exc < .5); % get prob of significancy for contra-preferring neurons

                %%% inh                
                inh_ipsiPref_prob_dataFromShflDist_eachDay{im}{iday} = inh_prob_dataROC_from_shflDist{im}{iday}(auc_tM1_inh > .5); % get prob of significancy for ipsi-preferring neurons
                inh_contraPref_prob_dataFromShflDist_eachDay{im}{iday} = inh_prob_dataROC_from_shflDist{im}{iday}(auc_tM1_inh < .5); % get prob of significancy for contra-preferring neurons

                %%%% 
                exc_fractSigTuned_ipsi_eachDay{im}(iday) = mean(exc_ipsiPref_prob_dataFromShflDist_eachDay{im}{iday} <= alpha); % fraction of Ns with AUC > .5 that are significantly tuned (comparing AUC with shuffled AUCs)
                exc_fractSigTuned_contra_eachDay{im}(iday) = mean(exc_contraPref_prob_dataFromShflDist_eachDay{im}{iday} <= alpha); % fraction of Ns with AUC < .5 that are significantly tuned (comparing AUC with shuffled AUCs)   
                inh_fractSigTuned_ipsi_eachDay{im}(iday) = mean(inh_ipsiPref_prob_dataFromShflDist_eachDay{im}{iday} <= alpha); 
                inh_fractSigTuned_contra_eachDay{im}(iday) = mean(inh_contraPref_prob_dataFromShflDist_eachDay{im}{iday} <= alpha);        

            end
        end
    end    
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Same as above, but pooled across days %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Pool across days: probability of data ROC coming from the same distribution as shuffled (assuming shuffled is a normal distribution with mu and sigma equal to mean and std of shuffled ROC values for each neuron).

exc_prob_dataROC_from_shflDist_eachMouseDaysPooled = cell(1, length(mice));
inh_prob_dataROC_from_shflDist_eachMouseDaysPooled = cell(1, length(mice));
exc_fractSigTuned = nan(1, length(mice));
inh_fractSigTuned = nan(1, length(mice));

for im = 1:length(mice)
    exc_prob_dataROC_from_shflDist_eachMouseDaysPooled{im} = cell2mat(exc_prob_dataROC_from_shflDist{im});
    inh_prob_dataROC_from_shflDist_eachMouseDaysPooled{im} = cell2mat(inh_prob_dataROC_from_shflDist{im});
    
    %%%% Set fractions of significantly tuned neurons 
    exc_fractSigTuned(im) = mean(exc_prob_dataROC_from_shflDist_eachMouseDaysPooled{im} <= alpha); % fraction of neurons that are significantly tuned (comparing AUC with shuffled AUCs)
    inh_fractSigTuned(im) = mean(inh_prob_dataROC_from_shflDist_eachMouseDaysPooled{im} <= alpha);    
end


%% Same as above, but pooled across days: Get probabilities (of data ROC coming from the same distribution as shuffled) for ipsi and contra preferring neurons separately (neurons of all days pooled)
%%%% Plots are made in choicePref_ROC_exc_inh_plotsAllMice

if doChoicePref==0

    exc_ipsiPref_prob_dataFromShflDist = cell(1, length(mice));
    exc_contraPref_prob_dataFromShflDist = cell(1, length(mice));
    inh_ipsiPref_prob_dataFromShflDist = cell(1, length(mice));
    inh_contraPref_prob_dataFromShflDist = cell(1, length(mice));
    
    exc_fractSigTuned_ipsi = nan(1, length(mice));
    exc_fractSigTuned_contra = nan(1, length(mice));
    inh_fractSigTuned_ipsi = nan(1, length(mice));
    inh_fractSigTuned_contra = nan(1, length(mice));
    
    exc_fractSigIpsiTuned = nan(1, length(mice));
    exc_fractSigContraTuned = nan(1, length(mice));
    inh_fractSigIpsiTuned = nan(1, length(mice));
    inh_fractSigContraTuned = nan(1, length(mice));

    for im = 1:length(mice)
        
        % take AUC at time -1
        auc_tM1_exc = exc_allNsDaysPooled_eachMouse{im}(nPreMin,:); % take AUC at time -1
        auc_tM1_exc = auc_tM1_exc(~isnan(auc_tM1_exc)); % take only valid days

        auc_tM1_inh = inh_allNsDaysPooled_eachMouse{im}(nPreMin,:); % take AUC at time -1
        auc_tM1_inh = auc_tM1_inh(~isnan(auc_tM1_inh)); % take only valid days
        
        
        %%%%%%%%%%%%% Out of sig choice-tuned neurons, what fraction are ipsi and what fraction are contra
        %%% exc
        aa = (exc_prob_dataROC_from_shflDist_eachMouseDaysPooled{im} <= alpha);
        exc_fractSigIpsiTuned(im) = mean(auc_tM1_exc(aa==1)>.5);
        exc_fractSigContraTuned(im) = mean(auc_tM1_exc(aa==1)<.5);

        %%% inh        
        aa = (inh_prob_dataROC_from_shflDist_eachMouseDaysPooled{im} <= alpha);
        inh_fractSigIpsiTuned(im) = mean(auc_tM1_inh(aa==1)>.5);
        inh_fractSigContraTuned(im) = mean(auc_tM1_inh(aa==1)<.5);    
        
        
        %%%%%%%%%%%%% Fraction of Ns with AUC >.5 (or <0.5) that are significantly tuned (comparing AUC with shuffled AUCs)
        %%% exc
        exc_ipsiPref_prob_dataFromShflDist{im} = exc_prob_dataROC_from_shflDist_eachMouseDaysPooled{im}(auc_tM1_exc > .5); % get prob of significancy for ipsi-preferring neurons (ie neurons whose AUC is > .5 but are not necessarily significantly tuned!)
        exc_contraPref_prob_dataFromShflDist{im} = exc_prob_dataROC_from_shflDist_eachMouseDaysPooled{im}(auc_tM1_exc < .5); % get prob of significancy for contra-preferring neurons

        %%% inh
        inh_ipsiPref_prob_dataFromShflDist{im} = inh_prob_dataROC_from_shflDist_eachMouseDaysPooled{im}(auc_tM1_inh > .5); % get prob of significancy for ipsi-preferring neurons
        inh_contraPref_prob_dataFromShflDist{im} = inh_prob_dataROC_from_shflDist_eachMouseDaysPooled{im}(auc_tM1_inh < .5); % get prob of significancy for contra-preferring neurons

        %%%% 
        exc_fractSigTuned_ipsi(im) = mean(exc_ipsiPref_prob_dataFromShflDist{im} <= alpha); % fraction of Ns with AUC > .5 that are significantly tuned (comparing AUC with shuffled AUCs)
        exc_fractSigTuned_contra(im) = mean(exc_contraPref_prob_dataFromShflDist{im} <= alpha); % fraction of Ns with AUC < .5 that are significantly tuned (comparing AUC with shuffled AUCs)   
        inh_fractSigTuned_ipsi(im) = mean(inh_ipsiPref_prob_dataFromShflDist{im} <= alpha); 
        inh_fractSigTuned_contra(im) = mean(inh_contraPref_prob_dataFromShflDist{im} <= alpha);        

    end
end

end


    


%%
no

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plots of each mouse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
if eachMouse_do_savefigs(1) % make plots
%     savefigs = eachMouse_do_savefigs(2);
    % call the script to make plots for each mouse
    choicePref_ROC_exc_inh_plotsEachMouse
end




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Summary plots of all mouse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if sumMice_do_savefigs(1) % make plots
%     savefigs = sumMice_do_savefigs(2);
    % call the script to make plots for each mouse
    choicePref_ROC_exc_inh_plotsAllMice
end


