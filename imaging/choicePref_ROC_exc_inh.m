%% Main script to do ROC analysis on individual neurons to decode choice.
% After this script, run choicePref_ROC_exc_inh_plots_setVars.m


%% Set vars below:

aveAllTimes = -1; % 1; % if 1, average X_avm across all time point; if -1, average it across [-100 300]ms (rel choice); before computing ROC
outcome2ana = 'corr'; % 'corr'; 'incorr'; '';
saveVars = 1; % if 1, vars will be saved.
doshfl = 1; % 1: shuffle trial labels; (you set the following 0 in the roc function: also do the chance version (ieset half of them to 0, other half to 1); 0: dont do either. If 1, in addition to doing roc on actual traces, we shuffle trial labels to get roc values for shuffled case as well.)

chAl = 1; 
initAl = 0;

setFRsOnly = 0; % only set FRs (not ROC vars)
downSampSpikes = 1; % downsample spike traces (non-overalapping moving average of 3 frames).
normX = 1; % after downsampling set max peak to 1. Makes sense to do, since traces before downsampling are normalized to max


%%

mice = {'fni16','fni17','fni18','fni19'};
thStimStrength = 0; % 2; % what stim strength you want to use for computing choice pref.
useEqualNumTrs = 0; % if true, equal number of trials for HR and LR will be used to compute ROC.
trialHistAnalysis = 0;
makeplots = 0;

if downSampSpikes    
    frameLength = 1000/30.9; % sec.
    regressBins = round(100/frameLength); % 100ms # set to nan if you don't want to downsample.
else
    regressBins = nan;
end
%{
dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/'; % dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Farzaneh_Gamal/ROC';
if setFRsOnly, dirn0 = fullfile(dirn0,'FR'); else dirn0 = fullfile(dirn0,'ROC'); end
%}
nowStr = datestr(now, 'yymmdd-HHMMSS');
zScoreX = 0; % % it doesnt make any difference, bc z scoring is a linear operation, so it doesnt change ROC values. %if 1, trace of each neuron (at each frame) will be z scored, ie mean of that neuron across all trials will be subtracted from each trial and then it will be divided by the std of the neuron across all trials. 
if zScoreX
    softNorm = 1; % soft normalziation : neurons with sd<thAct wont have too large values after normalization
    thAct = 5e-4; % it will be used for soft normalization #1e-5 # neurons whose average activity during ep is less than thAct will be called non-active and will be excluded.
    namz = '_zscoredX';
else namz = ''; 
end

if normX, nmd = '_norm2max'; else nmd = ''; end

if aveAllTimes==1
    namav = '_aveAllTimes'; 
elseif aveAllTimes==-1
    namav = '_aveSurrChoice';     
else
    namav = ''; 
end


%%
% im = 1;
for im = 1:length(mice)
    
    mouse = mice{im};
    fprintf('==============================  Mouse %s ==============================\n\n', mouse)
    
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
    % fni16:         % you use the ep of each day to compute ROC ave for setting ROC histograms.        %     days = {'150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'};
    % fni17:         % ep_ms = [800, 1100]; % window to compute ROC distributions ... for fni17 you used [800 1100] to compute svm which does not include choice... this is not the case for fni16.         %     days = {'151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};
    
    
    %% load alldata (of first day) to set the side that corresponds to HR stimulus.

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
%     fprintf('Total number of behavioral trials: %d\n', length(all_data))

    hrChoiceSide = all_data(1).highRateChoicePort;


    %% Loop over days for each mouse

    if setFRsOnly
        X_svm_all_alld_exc = cell(1, length(days));
        X_svm_all_alld_inh = cell(1, length(days));
    else
        choicePref_all_alld_exc = cell(1, length(days));
        choicePref_all_alld_inh = cell(1, length(days));
        choicePref_all_alld_allN = cell(1, length(days));
        choicePref_all_alld_uns = cell(1, length(days));
        choicePref_all_alld_exc_shfl = cell(1, length(days));
        choicePref_all_alld_inh_shfl = cell(1, length(days));
        choicePref_all_alld_allN_shfl = cell(1, length(days));
        choicePref_all_alld_uns_shfl = cell(1, length(days));
        choicePref_all_alld_exc_chance = cell(1, length(days));
        choicePref_all_alld_inh_chance = cell(1, length(days));
        choicePref_all_alld_allN_chance = cell(1, length(days));
        choicePref_all_alld_uns_chance = cell(1, length(days));
    end

    corr_ipsi_contra = nan(length(days),2); % number of correct ipsi (left) and contra (right) trials.
    eventI_ds_allDays = nan(1,length(days));
    eventI_allDays = nan(1,length(days));    
    ipsiTrs_allDays = cell(1, length(days));
    contraTrs_allDays = cell(1, length(days));
    
    
    %% 
    % iday = 1;
    for iday = 1:length(days)

        dn = simpleTokenize(days{iday}, '_');

        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));

        fprintf('\n_________________  %s, day %d/%d (%s, sessions %s)  _________________\n', mouse, iday, length(days), imagingFolder, dn{2})

        signalCh = 2; % because you get A from channel 2, I think this should be always 2.
        pnev2load = [];
        [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
        [pd, pnev_n] = fileparts(pnevFileName);
        moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
        postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));


        %% Set choiceVec0: animal's response side that is nan for unwanted trial outcomes. 
        
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
        end

    %         load(moreName, 'inhibitRois')
        load(moreName, 'inhibitRois_pix')
        inhibitRois = inhibitRois_pix;

%         load(postName, 'timeStimOnset', 'time1stSideTry')
%         ep_ms0 = [floor(nanmin(time1stSideTry-timeStimOnset))-30-300, floor(nanmin(time1stSideTry-timeStimOnset))-30];
%         fprintf('Training window: [%d %d] ms\n', ep_ms0(1), ep_ms0(2))
%         ep_all(iday,:) = ep_ms0;


        %% Set the spikes trace
        
        if trialHistAnalysis==0
            if chAl==1
                load(postName, 'firstSideTryAl')
                traces = firstSideTryAl.traces;
                times = firstSideTryAl.time;
                eventI = firstSideTryAl.eventI; 
                trsExcluded = isnan(squeeze(sum(sum(traces, 2),1))') + isnan(choiceVec0) ~= 0;            
                
            elseif initAl==1 % align on initiation tone
                load(postName, 'initToneAl')
                traces = initToneAl.traces;
                times = initToneAl.time;
                eventI = initToneAl.eventI;                 
                
                %%%%%%%% NOTE 1:
                % choiceVec is nan for trs2rmv and trials that mouse didn't make a choice.
                % outcomes is nan only for trs2rmv.
                % so here we use outcomes... instead of choiceVec
                % I dont think we care about choiceVec value when studying traces aligned on initTone... at least when setFrsOnly is 1, we should not care about this!
                
                % NOTE 2:
                % if you want to only analyze corr, or incorr trials, set
                % outcomes to nan in the section above, just how u do it
                % for choiceVec.
                trsExcluded = isnan(squeeze(sum(sum(traces, 2),1))') + isnan(outcomes) ~= 0; 
            end
%                 stimAl = stimAl_noEarlyDec;
    
    
            X_svm = traces(:,:,~trsExcluded);
            time_trace = times;     
        
        else
            load(postName, 'stimAl_allTrs')
            stimAl = stimAl_allTrs;            
        end

        eventI_allDays(iday) = eventI;


        %%% Set a copy of it before downsampling
        X_svmo = X_svm;
        fprintf('%d x %d x %d : size(X_svm)\n', size(X_svmo))

        
        %% Downsample X: average across multiple times (downsampling, not a moving average. we only average every regressBins points.)

        if isnan(regressBins)==0 % set to nan if you don't want to downsample

            %%%%%%%%%% set frames before frame0 (not including it)
            e = eventI-1;
            f = e - regressBins*floor(e/regressBins) + 1 : e; % 1st frame until 1 frame before frame0 (so that the total length is a multiplicaion of regressBins)
            x = X_svmo(f,:,:); % X_svmo including frames before frame0
            [T1, N1, C1] = size(x);
            tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames before frame0
            xdb = squeeze(mean(reshape(x, [regressBins, tt, N1, C1]), 1)); % downsampled X_svmo inclusing frames before frame0


            %%%%%%%%%%% set frames after frame0 (including it)
            lenPost = size(X_svmo,1) - (eventI-1);
            f = eventI : (eventI-1) + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins    
%             f = eventI+1 : eventI + regressBins * floor(lenPost/regressBins); % total length is a multiplicaion of regressBins    
            x = X_svmo(f,:,:); % X_svmo including frames after frame0
            [T1, N1, C1] = size(x);
            tt = floor(T1 / regressBins); % number of time points in the downsampled X including frames after frame0
            xda = squeeze(mean(reshape(x, [regressBins, tt, N1, C1]), 1)); % downsampled X_svmo inclusing frames after frame0


            %%%%%%%%%%% set the final downsampled X_svmo: concatenate downsampled X at frames before frame0, with x at frame0, and x at frames after frame0
            X_svm_d = cat(1, xdb, xda);    
%             X_svm_d = cat(1, xdb, X_svmo(eventI,:,:), xda);    
            X_svm = X_svm_d;
    %             print 'trace size--> original:',X_svmo.shape, 'downsampled:', X_svm_d.shape
            fprintf('%d x %d x %d : size(downsampled X_svm)\n', size(X_svm))

            
            %%%%%%%%%% set downsampled eventI
            eventI_ds = size(xdb,1)+1;
            
            
            %% Average across all time point
            
            if aveAllTimes==1 % average across all time points
                X_svm = mean(X_svm,1); % units x trials
            elseif aveAllTimes==-1 % average across [-200 300]ms relative to choice
                frs = eventI_ds-2 : eventI_ds+3;
                X_svm = mean(X_svm(frs,:,:),1); % units x trials
            end
           
            
            %% After downsampling normalize X_svm so each neuron's max is at 1 (you do this in matlab for S traces before downsampling... so it makes sense to again normalize the traces After downsampling so max peak is at 1)                

            if normX
                % find the max of each neurons across all trials and frames # max(X_svm.flatten())            
                m = max(X_svm, [], 3); % max across trials
                m = max(m, [], 1);  % max across frames            
                X_svm = bsxfun(@rdivide, X_svm, m);
            end
            
        else % don't downsample
            eventI_ds = nan;
        end

        eventI_ds_allDays(iday) = eventI_ds;
           
        
        %% Keep a copy of X_svm before z scoring
        
        if zScoreX
        
            X_svm00 = X_svm;

            %%% Center and normalize X: feature normalization and scaling: to remove effects related to scaling and bias of each neuron, we need to zscore data (i.e., make data mean 0 and variance 1 for each neuron) 

            % Normalize each frame separately (do soft normalization)
            X_svm_N = nan(size(X_svm00));
            meanX_fr = nan(size(X_svm00,1), size(X_svm00,2));
            stdX_fr = nan(size(X_svm00,1), size(X_svm00,2));
            for ifr = 1:size(X_svm00,1)
                m = squeeze(mean(X_svm00(ifr,:,:), 3)); % 1xneurons % average across trials
                s = squeeze(std(X_svm00(ifr,:,:),[], 3));   
                meanX_fr(ifr,:) = m; % frs x neurons
                stdX_fr(ifr,:) = s;       

                if softNorm==1 % soft normalziation : neurons with sd<thAct wont have too large values after normalization
                    s = s+thAct;     
                end

                X_svm_N(ifr,:,:) = bsxfun(@rdivide, bsxfun(@minus, squeeze(X_svm00(ifr,:,:))' , m) , s)';
            end

            X_svm = X_svm_N;

        end
        

        
        
        %% Set ipsiTrs and contraTrs, ie trials that the animal chose the ipsi or contra side 
        % needed for computing choice preference, 2*(auc-0.5), for each neuron at each frame.

        if trialHistAnalysis==0
            if strcmp(hrChoiceSide, 'L') % HR is left
                choseLeft = (choiceVec0==1); %(outcomes==1) & (allResp_HR_LR==1); % correct HR side --> left
                choseRight = (choiceVec0==0); %(outcomes==1) & (allResp_HR_LR==0); 

            else % LR is left
                choseLeft = (choiceVec0==0); %(outcomes==1) & (allResp_HR_LR==0);
                choseRight = (choiceVec0==1); %(outcomes==1) & (allResp_HR_LR==1); % correct HR side --> right                
            end

            
            if strcmp(outcome2ana, 'corr') || strcmp(outcome2ana, 'incorr') 
                % if a trial's stimrate is cb, it will be set to 0 (in both ipsiTrs and contraTrs).
                % I am actually not quite convinced we need to do this, but I guess it is better to make sure what we call corr or incorr does not include cb trials.
                ipsiTrs = choseLeft'  &  abs(stimrate-cb) > thStimStrength;
                contraTrs = choseRight'  &  abs(stimrate-cb) > thStimStrength;
            else
                ipsiTrs = choseLeft';
                contraTrs = choseRight';
            end
            
        else
            choseLeft = trialHistory.choiceVec0(:,3)==1;
            choseRight = trialHistory.choiceVec0(:,3)==0;

            ipsiTrs = choseLeft'; % &  abs(stimrate-cb) > thStimStrength);
            contraTrs = choseRight'; % &  abs(stimrate-cb) > thStimStrength);
        end

        fprintf('%d, %d: Number of left & right (stim diff > %d)\n', [sum(ipsiTrs), sum(contraTrs), thStimStrength])


        %% Now remove trsExcluded from ipsiTrs and contraTrs so their size match the traces size (X_svm)

        ipsiTrs = ipsiTrs(~trsExcluded);
        contraTrs = contraTrs(~trsExcluded);
        fprintf('%d, %d: Number of left and right after removing trsExcluded (stim diff > %d)\n', [sum(ipsiTrs), sum(contraTrs), thStimStrength])
        
        %{ 
        % below is true only if chAl=1; otherwise eg initAl=1, ipsi and
        contraTrs could be both 0, because the outcome was early decision,
        etc.
        if sum(sum([ipsiTrs, contraTrs],2)~=1)~=0
            s = stimrate(~trsExcluded); % then this trial must be of rate cb (16hx)
            if s(sum([ipsiTrs, contraTrs],2)~=1)~=cb
                error('somrthing wrong... a trial must be either ipsi or contra, it cannot be neither or both!')
            end
        end        
        %}
        corr_ipsi_contra(iday,:) = [sum(ipsiTrs) sum(contraTrs)]; % var name is not proper bc we may run the script with outcome2ana = 'incorr'... better name is "numTrs_ipsi_contra"
        ipsiTrs_allDays{iday} = ipsiTrs;
        contraTrs_allDays{iday} = contraTrs;
            
            
            
        %% Set firing rates for exc, inh neurons
        
        if setFRsOnly
            
            X_svm_all_alld_exc{iday} = X_svm(:,inhibitRois==0,:); % frames x excNeurons x trials
            X_svm_all_alld_inh{iday} = X_svm(:,inhibitRois==1,:); % frames x inhNeurons x trials        
            
        else % set ROC values for each neuron

            %% Do ROC analysis
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %% Now loop through exc, inh neurons to do ROC analysis

            for nt = 0:3 % neuron type (0:exc ; 1:inh)    

                if nt==0
                    disp('Analyzing excitatory neurons...')
                elseif nt==1
                    disp('Analyzing inhibitory neurons...')
                elseif nt==2
                    disp('Analyzing all neurons...')
                elseif nt==3
                    disp('Analyzing unsure neurons...')       
                end        

                %%% Set the neuron traces
                if nt==2
                    X_svm_now = X_svm;
                elseif nt==3
                    X_svm_now = X_svm(:, isnan(inhibitRois),:);
                else
                    X_svm_now = X_svm(:,inhibitRois==nt,:);
                end
                

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%  %%%%%%%%%%%%%%% DO ROC ANALYSIS %%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % Compute choicePref for each frame % for both actual and shuffled trial labels.
                % choicePref_all: frames x units. choicePref at each frame for each neuron
                
                % here we comute ChoicePref, below we compute auc from choicePref.
                % compute choice preference (2*(auc-0.5)) for each neuron at each frame.

                % ipsi is asigned 0 and contra is asigned 1 in "targets", so assumption is ipsi response is lower than contra. 
                % so auc>.5 (choicePref>0) happens when ipsi resp<contra, and auc<.5 (choicePref<0) happens when ipsi>contra.

                doChoicePref = 1; % if 1 we are interested in choicePref values; otherwise we want the AUC values. % otherwise we go with values of auc.                        
                [choicePref_all, choicePref_all_shfl, choicePref_all_chance] = ...
                    choicePref_ROC(X_svm_now, ipsiTrs, contraTrs, makeplots, eventI_ds, useEqualNumTrs, doChoicePref, doshfl); % frs x neurons


                %%
                if nt==0
                    choicePref_all_alld_exc{iday} = choicePref_all; % frames x neurons
                    choicePref_all_alld_exc_shfl{iday} = choicePref_all_shfl; % frames x neurons x samps
                    choicePref_all_alld_exc_chance{iday} = choicePref_all_chance;
                elseif nt==1
                    choicePref_all_alld_inh{iday} = choicePref_all; % frames x neurons
                    choicePref_all_alld_inh_shfl{iday} = choicePref_all_shfl; % frames x neurons x samps
                    choicePref_all_alld_inh_chance{iday} = choicePref_all_chance;
                elseif nt==2
                    choicePref_all_alld_allN{iday} = choicePref_all; % frames x neurons
                    choicePref_all_alld_allN_shfl{iday} = choicePref_all_shfl; % frames x neurons x samps
                    choicePref_all_alld_allN_chance{iday} = choicePref_all_chance;      
                elseif nt==3
                    choicePref_all_alld_uns{iday} = choicePref_all; % frames x neurons
                    choicePref_all_alld_uns_shfl{iday} = choicePref_all_shfl; % frames x neurons x samps
                    choicePref_all_alld_uns_chance{iday} = choicePref_all_chance;                       
                end        
            end    
        end
    end


    %% Save vars to a mat file
    
    if saveVars
        mouse = mice{im};    
        [~,~,d] = setImagingAnalysisNames(mouse, 'analysis', []);    % dirn = fullfile(dirn0fr, mouse);        d = fullfile(dirn, mouse); %dirn0         if ~exist(d, 'dir')            mkdir(d)        end

        if chAl, al = 'chAl'; 
        elseif initAl==1, al = 'initAl'; end % 'stAl'
        
        if strcmp(outcome2ana, 'corr')
            o2a = '_corr'; 
        elseif strcmp(outcome2ana, 'incorr')
            o2a = '_incorr';  
        else
            o2a = '_allOutcome';
        end        
        
%         if doshfl==-1, cn='_chance'; else cn=''; end
        if isnan(regressBins), dsn = '_noDownSamp'; else dsn = ''; end
        
        if setFRsOnly, fnam = sprintf('FR%s%s', dsn, nmd); else fnam = 'ROC'; end
            
        if trialHistAnalysis
            namv = sprintf('%s_prev_%s%s%s_stimstr%d%s_%s_%s.mat', fnam, al,o2a,namav,thStimStrength, namz, mouse, nowStr);
        else
            namv = sprintf('%s_curr_%s%s%s_stimstr%d%s_%s_%s.mat', fnam, al,o2a,namav,thStimStrength, namz, mouse, nowStr);
        end

        if setFRsOnly
            disp('saving FR vars for exc and inh neurons....')            
            save(fullfile(d,namv), 'X_svm_all_alld_exc', 'X_svm_all_alld_inh', 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'ipsiTrs_allDays', 'contraTrs_allDays')
        else
            disp('saving roc vars for exc and inh neurons....')            
%             save(fullfile(d,namv), 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'choicePref_all_alld_exc', 'choicePref_all_alld_inh', 'choicePref_all_alld_exc_shfl', 'choicePref_all_alld_inh_shfl', 'choicePref_all_alld_exc_chance', 'choicePref_all_alld_inh_chance')        
%             save(fullfile(d,namv), 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'choicePref_all_alld_exc', 'choicePref_all_alld_inh', 'choicePref_all_alld_exc_shfl', 'choicePref_all_alld_inh_shfl', 'choicePref_all_alld_exc_chance', 'choicePref_all_alld_inh_chance', 'choicePref_all_alld_allN', 'choicePref_all_alld_allN_shfl', 'choicePref_all_alld_allN_chance')
            save(fullfile(d,namv), 'corr_ipsi_contra', 'eventI_allDays', 'eventI_ds_allDays', 'choicePref_all_alld_exc', 'choicePref_all_alld_inh', 'choicePref_all_alld_exc_shfl', 'choicePref_all_alld_inh_shfl', 'choicePref_all_alld_exc_chance', 'choicePref_all_alld_inh_chance', 'choicePref_all_alld_allN', 'choicePref_all_alld_allN_shfl', 'choicePref_all_alld_allN_chance', 'choicePref_all_alld_uns', 'choicePref_all_alld_uns_shfl', 'choicePref_all_alld_uns_chance')
        end
    end
    
end




no

%% %%%%%% Load ROC vars and set vars required for plotting for each mouse %%%%%%
% go to this script to set vars (for saving and making plots)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

choicePref_ROC_exc_inh_plots_setVars.m
% choicePref_ROC_exc_inh_plotsEachMouse_setVars.m




%%
%{
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
%}

    
    
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

