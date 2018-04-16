% First run corr_excInh_setVars to get FRs
% Then run choicePref_ROC_exc_inh_plots_setVars to get ROCs (to set ipsi, contra tuned neurons) 
% Then run corr_excInh_plots

%% %%%%%% Load ROC vars and set vars required for plotting for each mouse %%%%%%

downSampSpikes = 1; %0; %1; % downsample spike traces (non-overalapping moving average of 3 frames).

alFR = 'chAl'; % 'initAl'; % the firing rate traces were aligned on what
outcome2ana = ''; %'corr';

fni18_rmvDay4 = 0; % if 1, remove 4th day of fni18.
doChoicePref = 0; %2; 
    % doChoicePref=0;  % use area under ROC curve % cares about ipsi bigger than contra or vice versa
    % doChoicePref=1;  % use choice pref = 2(AUC-.5) % cares about ipsi bigger than contra or vice versa
    % doChoicePref=2;  % Compute abs deviation of AUC from chance (.5); % doesnt care about ipsi bigger than contra or vice versa
normX = 1; % load FRs % after downsampling set max peak to 1. Makes sense to do, since traces before downsampling are normalized to max    
chAl = 1; % Is ROC is computed on choice-aligned traces?

thMinTrs = 10; % days with fewer than this number wont go into analysis      
mice = {'fni16','fni17','fni18','fni19'};
thStimStrength = 0;
plotchance = 0; % it didnt make almost any difference so go with shuffled not chance % if 0, look at shuffled (tr shuffled, if imbalance hr and lr, it would be different from chance); % if 1, plot chance (where 0s and 1s were manually made the same number). % redefine _shfl vars as _chance vars, bc we want to look at chance values



% eachMouse_do_savefigs = [1,1]; % Whether to make plots and to save them for each mouse
% sumMice_do_savefigs = [1,1]; % Whether to make plots and to save them for the summary of mice
if downSampSpikes    
    frameLength = 1000/30.9; % sec.
    regressBins = round(100/frameLength); % 100ms # set to nan if you don't want to downsample.
else
    regressBins = nan;
end

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
if zScoreX,     namz = '_zscoredX'; else,    namz = ''; end

if chAl
    al = 'chAl'; 
% elseif initAl==1
%     al = 'initAl'; 
end
        
if normX, nmd = '_norm2max'; else, nmd = ''; end

if strcmp(outcome2ana, 'corr')
    o2a = '_corr'; 
elseif strcmp(outcome2ana, 'incorr')
    o2a = '_incorr';  
else
    o2a = '_allOutcome';
end      

if isnan(regressBins), dsn = '_noDownSamp'; else dsn = ''; end


%%
nPreMin_allMice = nan(1, length(mice));
time_aligned_allMice = cell(1, length(mice));

nowStr_allMice = cell(1, length(mice));
mnTrNum_allMice = cell(1, length(mice));
days_allMice = cell(1, length(mice));
corr_ipsi_contra_allMice = cell(1, length(mice));

fr_exc_aligned_allMice = cell(1, length(mice));
fr_inh_aligned_allMice = cell(1, length(mice));

ipsiTrs_allDays_allMice = cell(1, length(mice));
contraTrs_allDays_allMice = cell(1, length(mice));
       

%%
% im = 1;
for im = 1:length(mice)

    mouse = mice{im};    
    [~,~,dirn] = setImagingAnalysisNames(mouse, 'analysis', []);    % dirn = fullfile(dirn0fr, mouse);
    
    
    %% Load FR vars

    namv = sprintf('FR%s%s_curr_%s%s_stimstr%d%s_%s_*.mat', dsn, nmd, alFR,o2a,thStimStrength,namz,mouse);    
    a = dir(fullfile(dirn,namv));
    a = a(end); % use the latest saved file
    namatfr = a.name;
    fprintf('%s\n',namatfr)    
    
    clear('X_svm_all_alld_exc', 'X_svm_all_alld_inh', 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'ipsiTrs_allDays', 'contraTrs_allDays')
    load(fullfile(dirn, namatfr), 'X_svm_all_alld_exc', 'X_svm_all_alld_inh', 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'ipsiTrs_allDays', 'contraTrs_allDays') % each cell is for a day and has size: frs x units x trials
    ipsiTrs_allDays_allMice{im} = ipsiTrs_allDays;
    contraTrs_allDays_allMice{im} = contraTrs_allDays;
    
    
    %% Set dir for loading ROC vars
    
%     dirn = fullfile(dirn0, mouse);
    namv = sprintf('ROC_curr_%s%s_stimstr%d%s_%s_*.mat', al,o2a,thStimStrength,namz,mouse);    
    a = dir(fullfile(dirn,namv));
    a = a(end); % use the latest saved file
    namatf = a.name;
    fprintf('%s\n',namatf)    
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
    
    if ~exist('corr_ipsi_contra', 'var')
        clear('corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays') %, 'choicePref_all_alld_exc', 'choicePref_all_alld_inh', 'choicePref_all_alld_exc_shfl', 'choicePref_all_alld_inh_shfl', 'choicePref_all_alld_exc_chance', 'choicePref_all_alld_inh_chance')
        load(fullfile(dirn, namatf), 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays') % , 'choicePref_all_alld_exc', 'choicePref_all_alld_inh', 'choicePref_all_alld_exc_shfl', 'choicePref_all_alld_inh_shfl', 'choicePref_all_alld_exc_chance', 'choicePref_all_alld_inh_chance')  
    end

%     if plotchance % redefine _shfl vars as _chance vars, bc we want to look at chance values
%         choicePref_all_alld_exc_shfl = choicePref_all_alld_exc_chance;
%         choicePref_all_alld_inh_shfl = choicePref_all_alld_inh_chance;
%     end
%     if isempty(choicePref_all_alld_inh_shfl{1})
%         doshfl = 0;
%     else
%         doshfl = 1;
%     end
    
    % get rid of the bad day of fni18
    if fni18_rmvDay4 && im==imfni18 %&& iday==4 % set this day to nan
        corr_ipsi_contra(4,:) = thMinTrs-1; % set it to a value lower than thMinTrs, so it gets removed!
    end
    
%     nsamps = size(choicePref_all_alld_exc_shfl{1},3);        
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
        nPost(iday) = size(X_svm_all_alld_inh{iday},1) - eventI_ds_allDays(iday);
    end
    nPostMin = min(nPost);

    nPreMin = min(eventI_ds_allDays)-1;
%     disp([nPreMin, nPostMin])


    %% Set time_aligned for the aligned trace for each mouse

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
%     size(time_aligned)
            

    %% Align FRs of all days on the common eventI; exclude days with too few trials.
    %%% REMEMBER: you are doing this weird thing for convenience: 
        % for days with few trials, we use nans for arbitrarily 3 neurons and 3 trials!
            
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
    
    
    
    %% Keep vars of all mice

    nPreMin_allMice(im) = nPreMin;
    time_aligned_allMice{im} = time_aligned;
    
    
    fr_exc_aligned_allMice{im} = fr_exc_aligned;
    fr_inh_aligned_allMice{im} = fr_inh_aligned;
    
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

fr_exc_al_allMice = cell(1, length(mice));
fr_inh_al_allMice = cell(1, length(mice));

for im = 1:length(mice)
    
    ev = nPreMin_allMice(im)+1; % eventI of the day-aligned traces for mouse im    
    for iday = 1:length(fr_exc_aligned_allMice{im})        
        fr_exc_al_allMice{im}{iday} = fr_exc_aligned_allMice{im}{iday}(ev - nPreMin  :  ev + nPostMin, :, :); % nAlignedFrs (across mice) x neurons
        fr_inh_al_allMice{im}{iday} = fr_inh_aligned_allMice{im}{iday}(ev - nPreMin  :  ev + nPostMin, :, :); % nAlignedFrs (across mice) x neurons
    end
end



%% Number of valid days for each mouse

numDaysAll = nan(1,length(mice));
numDaysGood = nan(1,length(mice));
for im = 1:length(mice)
    numDaysAll(im) = length(fr_exc_al_allMice{im});
    numDaysGood(im) = sum(mnTrNum_allMice{im}>=thMinTrs); % sum(~isnan(aveexc_allMice{im}(1,:)));
end
numDaysAll
numDaysGood


%% So the vars (not to get confused with ROC vars)

nPreMin_allMice_fr = nPreMin_allMice;
time_aligned_allMice_fr = time_aligned_allMice;
nPreMin_fr = min(nPreMin_allMice_fr);

nowStr_allMice_fr = nowStr_allMice;
mnTrNum_allMice_fr = mnTrNum_allMice;
days_allMice_fr = days_allMice;
corr_ipsi_contra_allMice_fr = corr_ipsi_contra_allMice;

numDaysAll_fr = numDaysAll;
nDaysGood_fr = numDaysGood;

