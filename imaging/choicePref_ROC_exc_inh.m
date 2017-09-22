saveVars = 1; % if 1, vars will be saved.
savefigs = 0;

outcome2ana = 'corr';
chAl = 1;

thStimStrength = 0; % 2; % what stim strength you want to use for computing choice pref.
useEqualNumTrs = 0; % if true, equal number of trials for HR and LR will be used to compute ROC.
trialHistAnalysis = 0;
makeplots = 0;
frameLength = 1000/30.9; % sec.
regressBins = round(100/frameLength); % 100ms # set to nan if you don't want to downsample.

for mice = {'fni16','fni17','fni18','fni19'}
    
    mouse = mice{1};

    %%% Set days for each mouse

    if strcmp(mouse, 'fni16')
        % you use the ep of each day to compute ROC ave for setting ROC histograms.    
    %     days = {'150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'};
        days = {'150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'}; %'150914_1-2' : don't analyze!

    elseif strcmp(mouse, 'fni17')
        ep_ms = [800, 1100]; % window to compute ROC distributions ... for fni17 you used [800 1100] to compute svm which does not include choice... this is not the case for fni16.    
    %     days = {'151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};
        days = {'150814_1', '150817_1', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};

    elseif strcmp(mouse, 'fni18')
        days = {'151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'}; % alldays

    elseif strcmp(mouse, 'fni19')    
        days = {'150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'};
    end


    %% load alldata to set hr choice side 

    dn = simpleTokenize(days{1}, '_');
    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
    
    % load(imfilename, 'all_data'), all_data = all_data(1:end-1);   %from the imaging file.  % alldata = removeBegEndTrs(alldata, thbeg);
    % use the following if you want to go with the alldata of the behavior folder
    % set filenames
    [alldata_fileNames, ~] = setBehavFileNames(mouse, {datestr(datenum(imagingFolder, 'yymmdd'))});
    % sort it
    [~,fn] = fileparts(alldata_fileNames{1});
    a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
    [~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
    alldata_fileNames = alldata_fileNames(isf);
    % load the one corresponding to mdffilenumber.
    [all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber)); % , defaultHelpedTrs, saveHelpedTrs); % it removes the last trial too.
    fprintf('Total number of behavioral trials: %d\n', length(all_data))

    hrChoiceSide = all_data(1).highRateChoicePort;



    %%

    choicePref_all_alld_exc = cell(1, length(days));
    choicePref_all_alld_inh = cell(1, length(days));

    corr_ipsi_contra = nan(length(days),2); % number of correct ipsi (left) and contra (right) trials.
    eventI_ds_allDays = nan(1,length(days));
    eventI_allDays = nan(1,length(days));
    ep_all = nan(length(days),2);

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


        %%
        if trialHistAnalysis==1
            load(postName, 'trialHistory')            
        else
            load(postName, 'outcomes', 'allResp_HR_LR', 'stimrate', 'cb')

            choiceVec0 = allResp_HR_LR;  % trials x 1;  1 for HR choice, 0 for LR choice. % choice of the current trial.    

            if strcmp(outcome2ana, 'corr')
                choiceVec0(outcomes~=1) = nan; % analyze only correct trials.
            elseif strcmp(outcome2ana, 'incorr')
                choiceVec0(outcomes~=0) = nan; % analyze only incorrect trials.   
            end
    %             choiceVec0(~str2ana) = nan;
        end

    %         load(moreName, 'inhibitRois')
        load(moreName, 'inhibitRois_pix')
        inhibitRois = inhibitRois_pix;


        %%
        load(postName, 'timeStimOnset', 'time1stSideTry')

        ep_ms0 = [floor(nanmin(time1stSideTry-timeStimOnset))-30-300, floor(nanmin(time1stSideTry-timeStimOnset))-30];
        fprintf('Training window: [%d %d] ms\n', ep_ms0(1), ep_ms0(2))

        ep_all(iday,:) = ep_ms0;


        %%
        if trialHistAnalysis==0
            if chAl==1
                load(postName, 'firstSideTryAl')
                traces_al_1stSide = firstSideTryAl.traces;
                time_aligned_1stSide = firstSideTryAl.time;
                eventI = firstSideTryAl.eventI; 

                trsExcluded = isnan(squeeze(sum(sum(traces_al_1stSide, 2),1))') + isnan(choiceVec0) ~= 0;

                X_svm = traces_al_1stSide(:,:,~trsExcluded);
                time_trace = time_aligned_1stSide;     

    %             else
    %                 load(postName, 'stimAl_noEarlyDec')
            end
    %             stimAl = stimAl_noEarlyDec;
        else
            load(postName, 'stimAl_allTrs')
            stimAl = stimAl_allTrs;            
        end

        eventI_allDays(iday) = eventI;


        %%
        X_svmo = X_svm;
        size(X_svmo)


        %% Downsample X: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)

        if isnan(regressBins)==0 % set to nan if you don't want to downsample.

            % set frames before frame0 (not including it)
            e = eventI-1;
            f = e - regressBins*floor(e/regressBins) + 1 : e; % 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
            x = X_svmo(f,:,:); % X_svmo including frames before frame0
            [T1, N1, C1] = size(x);
            tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames before frame0
            xdb = squeeze(mean(reshape(x, [regressBins, tt, N1, C1]), 1)); % downsampled X_svmo inclusing frames before frame0




            % set frames after frame0 (not including it)
            lenPost = size(X_svmo,1) - eventI;
            f = eventI+1 : eventI + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins    
            x = X_svmo(f,:,:); % X_svmo including frames after frame0
            [T1, N1, C1] = size(x);
            tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames after frame0
            xda = squeeze(mean(reshape(x, [regressBins, tt, N1, C1]), 1)); % downsampled X_svmo inclusing frames after frame0

            % set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frame0, and x at frames after frame0
            X_svm_d = cat(1, xdb, X_svmo(eventI,:,:), xda);    

            X_svm = X_svm_d;
    %             print 'trace size--> original:',X_svmo.shape, 'downsampled:', X_svm_d.shape
            size(X_svm)



            % set downsampled eventI
            eventI_ds = size(xdb,1)+1;
        end

        eventI_ds_allDays(iday) = eventI_ds;


        %% Set vars for computing choice preference, 2*(auc-0.5), for each neuron at each frame.

        % choicePref_ROC

        % set ipsi and contra trials
        if trialHistAnalysis==0
            if strcmp(hrChoiceSide, 'L') % HR is left
                correctL = (outcomes==1) & (allResp_HR_LR==1); % correct HR side --> left
                correctR = (outcomes==1) & (allResp_HR_LR==0); 

            else % LR is left
                correctL = (outcomes==1) & (allResp_HR_LR==0);
                correctR = (outcomes==1) & (allResp_HR_LR==1); % correct HR side --> right                
            end

            % what stim rates to use?
            ipsiTrs = correctL'  &  abs(stimrate-cb) > thStimStrength;
            contraTrs = correctR'  &  abs(stimrate-cb) > thStimStrength;

        else
            correctL = trialHistory.choiceVec0(:,3)==1;
            correctR = trialHistory.choiceVec0(:,3)==0;

            ipsiTrs = correctL'; % &  abs(stimrate-cb) > thStimStrength);
            contraTrs = correctR'; % &  abs(stimrate-cb) > thStimStrength);
        end

        fprintf('Num corrL and corrR (stim diff > %d): %d,  %d\n', [thStimStrength, sum(ipsiTrs) sum(contraTrs)])

        %%%
    %         X_svm_now = X_svm_now(:,:,~trsExcluded); % already done above
        ipsiTrs = ipsiTrs(~trsExcluded);
        contraTrs = contraTrs(~trsExcluded);
        fprintf('After removing trsExcluded: Num corrL and corrR (stim diff > %d): %d,  %d\n', [thStimStrength, sum(ipsiTrs) sum(contraTrs)])

        corr_ipsi_contra(iday,:) = [sum(ipsiTrs) sum(contraTrs)];

    % end
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %% Now loop through exc, inh neurons to do ROC analysis

    %     traces_al_sm0 = X_svm; %stimAl.traces;
    %     eventI_stimOn = eventI_ds; %stimAl.eventI;        
        for nt = 0:1 % neuron type (0:exc ; 1:inh)    

            if nt==0
                disp('Analyzing excitatory neurons.')
        %         choicePref_all_alld_exc = cell(1, length(days));

            elseif nt==1
                disp('Analyzing inhibitory neurons.')
        %         choicePref_all_alld_inh = cell(1, length(days));
            end        

            %% Set the neuron traces

            X_svm_now = X_svm(:,inhibitRois==nt,:);
    %         trsExcluded = isnan(squeeze(mean(mean(X_svm_now,1),2)));


            %% Compute choicePref for each frame

            % dont worry about doChoicePref, later below you compute auc from choicePref
    %         doChoicePref = 0; % if 1 we are interested in choicePref values; otherwise we want the AUC values. % otherwise we go with values of auc.

            % choicePref_all: frames x units. choicePref at each frame for each neuron
            choicePref_all = choicePref_ROC(X_svm_now, ipsiTrs, contraTrs, makeplots, eventI_ds, useEqualNumTrs);


            %%
            if nt==0
                choicePref_all_alld_exc{iday} = choicePref_all; % frames x neurons
            elseif nt==1
                choicePref_all_alld_inh{iday} = choicePref_all; % frames x neurons
            end        

        end    
    end

%     mean(ep_all)
%     min(ep_all)
%     max(ep_all)


    %% Save vars
    
    if saveVars

        d = fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse);
        if ~exist(d, 'dir')
            mkdir(d)
        end

        if chAl==1, al = 'chAl'; else, al = 'stAl'; end        
        if strcmp(outcome2ana, 'corr'), o2a = '_corr'; else, o2a = '';  end
        nowStr = datestr(now, 'yymmdd-HHMMSS');
        
        if trialHistAnalysis
            namv = sprintf('ROC_prev_%s%s_stimstr%d_%s_%s.mat', al,o2a,thStimStrength, mouse, nowStr);
        else
            namv = sprintf('ROC_curr_%s%s_stimstr%d_%s_%s.mat', al,o2a,thStimStrength, mouse, nowStr);
        end

        if exist(namv, 'file')==2
            error('File already exists... do you want to over-write it?')
        else
            disp('saving roc vars for exc and inh neurons....')
            
            save(fullfile(d,namv), 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'choicePref_all_alld_exc', 'choicePref_all_alld_inh')
        end
    end
end


%% PLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cod = get(groot,'defaultAxesColorOrder'); % default value
set(groot,'defaultAxesColorOrder',cod)
 

%% 
savefigs = 0;
doChoicePref = 0;
thMinTrs = 10; % days with fewer than this number wont go into analysis      
    
chAl = 1;
outcome2ana = 'corr';
thStimStrength = 0;
frameLength = 1000/30.9; % sec.
regressBins = round(100/frameLength); % 100ms # set to nan if you don't want to downsample.
mice = {'fni16','fni17','fni18','fni19'};

choicePref_exc_aligned_allMice = cell(1, length(mice));
choicePref_inh_aligned_allMice = cell(1, length(mice));
nPreMin_allMice = nan(1, length(mice));

for im = 1:length(mice)

    %%% Set days for each mouse
    mouse = mice{im};

    if strcmp(mouse, 'fni16')
        % you use the ep of each day to compute ROC ave for setting ROC histograms.    
    %     days = {'150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'};
        days = {'150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'}; %'150914_1-2' : don't analyze!

    elseif strcmp(mouse, 'fni17')
        ep_ms = [800, 1100]; % window to compute ROC distributions ... for fni17 you used [800 1100] to compute svm which does not include choice... this is not the case for fni16.    
    %     days = {'151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};
        days = {'150814_1', '150817_1', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};

    elseif strcmp(mouse, 'fni18')
        days = {'151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'}; % alldays

    elseif strcmp(mouse, 'fni19')    
        days = {'150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'};
    end
    
    
    %% Load roc vars
    
    d = fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse);
    if chAl==1, al = 'chAl'; else, al = 'stAl'; end        
    if strcmp(outcome2ana, 'corr'), o2a = '_corr'; else, o2a = '';  end    
    namv = sprintf('ROC_curr_%s%s_stimstr%d_%s_*.mat', al,o2a,thStimStrength,mouse);    
    a = dir(fullfile(d,namv));
    fprintf('%s\n',a.name)    

    load(fullfile(d,a.name), 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'choicePref_all_alld_exc', 'choicePref_all_alld_inh')  

    % min number of trials of the 2 class
    mnTrNum = min(corr_ipsi_contra,[],2);
    

    %% Prep to align all traces on common eventI

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


    %% Set time_aligned

    totLen = nPreMin + nPostMin +1;
    eventI = min(eventI_allDays);

    time_trace = frameLength * ((1 : ceil(regressBins*totLen)) - eventI); % time_trace = time_aligned_1stSide

    % set frames before frame0 (not including it)
    e = eventI-1;
    f = e - regressBins*floor(e/regressBins) + 1 : e; % 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
    x = time_trace(f); % time_trace including frames before frame0
    T1 = length(x);
    tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames before frame0
    xdb = squeeze(mean(reshape(x, [regressBins, tt]), 1)); % downsampled time_trace inclusing frames before frame0


    % set frames after frame0 (not including it)
    lenPost = length(time_trace) - eventI;
    f = eventI+1 : eventI + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins    
    x = time_trace(f); % time_trace including frames after frame0
    T1 = length(x);
    tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames after frame0
    xda = squeeze(mean(reshape(x, [regressBins, tt]), 1)); % downsampled time_trace inclusing frames after frame0

    % set the final downsampled time_trace: concatenate downsampled X at frames before frame0, with x at frame0, and x at frames after frame0
    time_trace_d = cat(2, xdb, 0, xda);    
    time_aligned = time_trace_d(1:totLen);
    %{
    a = -frameLength * (0:nPreMin); a = a(end:-1:1);
    b = frameLength * (1:nPostMin);
    time_aligned = [a,b];
    %}
    size(time_aligned)


    %% Align choice pref traces of all days on the common eventI
    
    choicePref_exc_aligned = cell(1,length(days)); % each cell: nFrames_aligned x neurons
    choicePref_inh_aligned = cell(1,length(days)); % nan(nPreMin + nPostMin + 1, length(days));
    % thMinTrs = 0;
    for iday = 1:length(days)
        if mnTrNum(iday,:) >= thMinTrs            
            choicePref_exc_aligned{iday} = -choicePref_all_alld_exc{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :); % now ipsi is positive bc we do 1 minus, in the original model contra is positive
            choicePref_inh_aligned{iday} = -choicePref_all_alld_inh{iday}(eventI_ds_allDays(iday) - nPreMin  :  eventI_ds_allDays(iday) + nPostMin, :);
            if doChoicePref==0  % use area under ROC curve
                % choice pref = 2*(auc-.5)
                choicePref_exc_aligned{iday} = (0.5 + choicePref_exc_aligned{iday}/2); % now ipsi auc will be above 0.5 bc we do - above
                choicePref_inh_aligned{iday} = (0.5 + choicePref_inh_aligned{iday}/2); % frames x neurons
            end
        else
            choicePref_exc_aligned{iday} = nan(nPreMin + nPostMin + 1 , 3); % set to nan so number of neurons doesnt matter... I just picked 3.
            choicePref_inh_aligned{iday} = choicePref_exc_aligned{iday};
        end
    end

    
    %% Keep vars of all mice
    
    choicePref_exc_aligned_allMice{im} = choicePref_exc_aligned;
    choicePref_inh_aligned_allMice{im} = choicePref_inh_aligned;
    nPreMin_allMice(im) = nPreMin;
    
    
    %% Average AUC across neurons for each day and each frame

    aveexc = cellfun(@(x)mean(x,2), choicePref_exc_aligned, 'uniformoutput',0); % average across neurons
    aveexc = cell2mat(aveexc); % frs x days
    seexc = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_exc_aligned, 'uniformoutput',0); % standard error across neurons
    seexc = cell2mat(seexc); % frs x days    

    aveinh = cellfun(@(x)mean(x,2), choicePref_inh_aligned, 'uniformoutput',0);
    aveinh = cell2mat(aveinh);
    seinh = cellfun(@(x)std(x,[],2)/sqrt(size(x,2)), choicePref_inh_aligned, 'uniformoutput',0);
    seinh = cell2mat(seinh); % frs x days        

    
    %% Plots of ave-neuron AUC    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Plot hist of ROC values, averaged across neurons for each session, pool all session and all days, compare exc and inh
    
%     figure; hold on;
%     histogram(aveexc);
%     histogram(aveinh);

    y1 = aveexc(:);
    y2 = aveinh(:);
    xlab = 'ROC AUC';
    ylab = 'Fraction days*frames';
    leg = {'exc','inh'};

    plotHist(y1,y2,xlab,ylab,leg)    


    if savefigs
        cd(fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse))
        savefig(fullfile(d, 'ROC_curr_chAl_excInh_dist_aveNeurs_frsDaysPooled.fig'))
    end
    
    % run ttest across days for each frame
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
    
    

    %% For each day compare exc and inh ROC across time

    figure('name', 'ipsi>contra; numTrs (ipsi contra)'); 
    ha = tight_subplot(ceil(size(aveexc,2)/10), 10, [.04 .04],[.03 .03],[.03 .03]);
    for iday = 1:size(aveexc,2)        
        if mnTrNum(iday) >= thMinTrs             
%             subplot(ceil(size(aveexc,2)/10), 10, iday)                                    
            [h1] = boundedline(ha(iday), time_aligned, aveexc(:,iday)', seexc(:,iday)', 'b', 'alpha', 'nan', 'remove');
            [h2] = boundedline(ha(iday), time_aligned, aveinh(:,iday)', seinh(:,iday)', 'r', 'alpha', 'nan', 'remove');
            
            hold(ha(iday),'on')
            yl = get(ha(iday),'ylim');
            plot(ha(iday), [0, 0], [yl(1), yl(2)], 'k:')
            plot(ha(iday), [time_aligned(1), time_aligned(end)], [.5,.5], 'k:')
            ylim(ha(iday), yl)
            da = days{iday};
            s = sprintf('%s (%d %d)', da(1:6), corr_ipsi_contra(iday,1), corr_ipsi_contra(iday,2));
            title(ha(iday), s)            
        end
    end
%     subplot(ceil(size(aveexc,2)/10), 10, 1)
    xlabel(ha(1), 'Time')
    ylabel(ha(1), 'ROC')
    legend(ha(1), [h1,h2], 'exc','inh')
    

    if savefigs
%         cd(fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse))
        savefig(fullfile(d, 'ROC_curr_chAl_excInh_timeCourse_aveNeurs_eachDay.fig'))
    end
     
    
    %% learning figure: For each frame, compare exc/inh neuron-averaged ROC across days
    
    figure(); 
    ha = tight_subplot(ceil((size(aveexc,1)+1)/4), 4, [.04 .04],[.03 .03],[.03 .03]);
%     subplot(ceil((size(aveexc,1)+1)/4), 4, 1)
    h = plot(ha(1), corr_ipsi_contra);
    legend(h, 'ipsi', 'contra')
    xlabel(ha(1), 'days')
    ylabel(ha(1), 'Num trials')
    
    for fr = 1:size(aveexc,1)        
%         subplot(ceil((size(aveexc,1)+1)/4), 4, fr+1)        
        [h1] = boundedline(ha(fr+1), 1:size(aveexc,2), aveexc(fr,:), seexc(fr,:), 'b', 'alpha', 'nan', 'remove');
        [h2] = boundedline(ha(fr+1), 1:size(aveinh,2), aveinh(fr,:), seinh(fr,:), 'r', 'alpha', 'nan', 'remove');
        hold(ha(fr+1),'on')
        plot(ha(fr+1), [0,length(days)+1], [.5,.5], 'k:')
        title(ha(fr+1), sprintf('%d ms',round(time_aligned(fr))))
    end
%     subplot(ceil((size(aveexc,1)+1)/4), 4, 2)
    xlabel(ha(2), 'days')
    ylabel(ha(2), 'ROC')
    legend(ha(2), [h1,h2], 'exc','inh')
    title(ha(2), sprintf('time rel2 choice = %d ms',round(time_aligned(1))))
    
    if savefigs
%         cd(fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse))
        savefig(fullfile(d, 'ROC_curr_chAl_excInh_trainingDays_aveNeurs_eachFrame.fig'))
    end
    
    
    
    %% Superimpose all days ROC vs. frame
  

    co = jet(size(aveexc,2));
    set(groot,'defaultAxesColorOrder',co)
    
    figure
    subplot(211)
    plot(time_aligned, aveexc)
    hold on
    plot([time_aligned(1),time_aligned(end)], [.5,.5], 'k:')
    yl = get(gca,'ylim');
    plot([0,0], yl, 'k:')
    ylim(yl)
    title('exc')
    legend(days)
    xlabel('time')
    ylabel('ROC')
    
    subplot(212)
    plot(time_aligned, aveinh)
    hold on
    plot([time_aligned(1),time_aligned(end)], [.5,.5], 'k:')
    yl = get(gca,'ylim');
    plot([0,0], yl, 'k:')
    ylim(yl)    
    title('inh')
    
%     figure
%     imagesc(aveexc)

    if savefigs
%         cd(fullfile('/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC', mouse))
        savefig(fullfile(d, 'ROC_curr_chAl_excInh_allDaysSup_timeCourse_aveNeurs.fig'))
    end
    
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Finish below, plotHist....
        
    
    %% Pool AUC of all neurons of all days

    excall = cell2mat(choicePref_exc_aligned); % aligned_frs x sum_exc_neurons
    inhall = cell2mat(choicePref_inh_aligned); % aligned_frs x sum_inh_neurons
    size(excall), size(inhall)
    
    figure; hold on;
    histogram(excall)%, 'Normalization', 'probability');
    histogram(inhall)%, 'Normalization', 'probability')
    
    h = ttest2(excall', inhall');
    hh = h;
    hh(h==0) = nan;


    %% Plots

    figure('position',[10   556   792   383]);
    % average and se across days; each day is already averaged across neurons.
    subplot(121); hold on
    h1 = boundedline(time_aligned, nanmean(aveexc,2), nanstd(aveexc,0,2)/sqrt(size(aveexc,2)), 'b', 'alpha');
    h2 = boundedline(time_aligned, nanmean(aveinh,2), nanstd(aveinh,0,2)/sqrt(size(aveinh,2)), 'r', 'alpha');
    a = get(gca, 'ylim');
    plot(time_aligned, hh0*(a(2)-.05*diff(a)), 'k')
    legend([h1,h2], 'Excitatory', 'Inhibitory')
    xlabel('Time since choice onset (ms)')
    ylabel('ROC AUC')
    title('mean+se days')

    %%%
    subplot(122); hold on
    h1 = boundedline(time_aligned, nanmean(excall,2), nanstd(excall,0,2)/sqrt(size(excall,2)), 'b', 'alpha');
    h2 = boundedline(time_aligned, nanmean(inhall,2), nanstd(inhall,0,2)/sqrt(size(inhall,2)), 'r', 'alpha');
    a = get(gca, 'ylim');
    plot(time_aligned, hh*(a(2)-.05*diff(a)), 'k')
    legend([h1,h2], 'Excitatory', 'Inhibitory')
    if chAl==1
        xlabel('Time since choice onset (ms)')
    else
        xlabel('Time since stim onset (ms)')
    end
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

end

    
    
%%
%{
% This part is for roc hists
if trialHistAnalysis==0 && strcmp(mouse, 'fni16')
    cprintf('blue', 'Using variable ep to compute average ROC for setting histograms!\n')
    % For fni16 (that you used variable epochs for training SVM) compute ROC
    % average in ep of each day instead of during a fixed window
    % Average of ROC values during choiceTime-300:choiceTime for each neuron in each day
    exc_ep = [];
    inh_ep = [];
    for iday = 1:length(days)
        epnow = eventI_ds_allDays(iday) + ceil(ep_all(iday,:)/frameLength);

        exc_ep = [exc_ep, mean(choicePref_exc_aligned{iday}(epnow(1):epnow(2), :), 1)];
        inh_ep = [inh_ep, mean(choicePref_inh_aligned{iday}(epnow(1):epnow(2), :), 1)];
    end
end
%}



%%

