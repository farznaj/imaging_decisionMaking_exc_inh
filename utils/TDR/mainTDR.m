%% Stability of subspaces (variance and stimulus, choice of TDR)


%%
trialHistAnalysis = 0; %1;
nt = 1;


frameLength = 1000/30.9; % sec.
makeplots = 0;
useEqualNumTrs = 0; % if true, equal number of trials for HR and LR will be used to compute ROC.
thStimStrength = 0; % 2; % what stim strength you want to use for computing choice pref.
doplots = 0;

% the following are needed for setting stim-aligned traces.
ep_ms = [800 1300]; %[700 900]; % make sure no trial includes go tone before the end of ep % ep_ms will be used for setting X (trials x neurons, average of neural activity in window ep relative to stimulus onset).
th_stim_dur = 800; % min stim duration to include a trial in timeStimOnset
% the following are needed for setting X, Y, TrsExcluded, NsExcluded
% trialHistAnalysis = 0;
outcome2ana = 'corr'; % only used if trialHistAnalysis is 0; % '', corr', 'incorr' # trials to use for SVM training (all, correct or incorrect trials)
strength2ana = 'all'; % only used if trialHistAnalysis is 0; % 'all', easy', 'medium', 'hard' % What stim strength to use for training?
thAct = 5e-4; % 1e-4; % quantile(spikeAveEpAveTrs, .1);
thTrsWithSpike = 1; % 3; % 1; % ceil(thMinFractTrs * size(spikeAveEp0,1)); % 30  % remove neurons with activity in <thSpTr trials.

    
mouse = 'fni17';
days = {'151102_1-2', '151101_1', '151029_2-3', '151028_1-2-3', '151027_2', '151026_1', ...
    '151023_1', '151022_1-2', '151021_1', '151020_1-2', '151019_1-2', '151016_1', ...
    '151015_1', '151014_1', '151013_1-2', '151012_1-2-3', '151010_1', '151008_1', '151007_1'};


%%
if nt==0
    aIx_all_alld_exc = cell(1, length(days));
    angle_all_alld_exc = cell(1, length(days));
elseif nt==1
    aIx_all_alld_inh = cell(1, length(days));
    angle_all_alld_inh = cell(1, length(days));
end


for iday = 1:length(days)
    
    disp('__________________________________________________________________')
    dn = simpleTokenize(days{iday}, '_');
    
    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
    
    fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})    
    
    
    %% Load stimrate and set the following variables
    
    % traces_al_stim: frames x units x trials; stimulus aligned traces. Only for active neurons and valid (non-nan) trials.
    % time_aligned_stim: 1 x frames; time points for non-filtered trace.
    % X: trials x neurons; includes average of window ep (defined in SVM codes) for each neuron at each trial.
    % Y: trials x 1; animal's choice on the current trial (0: LR, 1:HR)
    
    % stimrate: trials x 1; stimulus rate of each trial.
    
    
    signalCh = 2;
    pnev2load = []; %7 %4 % what pnev file to load (index based on sort from the latest pnev vile). Set [] to load the latest one.
    postNProvided = 1; % whether the directory cotains the postFile or not. if it does, then mat file names will be set using postFile, otherwise using pnevFile.
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load, postNProvided);
    [pd, pnev_n] = fileparts(pnevFileName);
    postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
    moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
    
    load(postName, 'stimrate', 'time1stSideTry', 'timeCommitCL_CR_Gotone', 'timeStimOnset', 'timeStimOffset', 'outcomes', 'allResp_HR_LR', 'cb')
    load(moreName, 'inhibitRois')
    if trialHistAnalysis==1
        load(postName, 'trialHistory','stimAl_allTrs')
    else
        load(postName, 'stimAl_noEarlyDec') %'outcomes', 'allResp_HR_LR', 'stimrate', 'cb','stimAl_noEarlyDec')
        stimAl_allTrs = stimAl_noEarlyDec;
    end
    % load('SVM_151029_003_ch2-PnevPanResults-160426-191859.mat')
    
    %%%%%%% Use stricter measures to exclude some neurons from stimAl traces
    % newBadROIs = findBadROIs_strict(mouse, imagingFolder, mdfFileNumber);
    % stimAl_allTrs.traces = stimAl_allTrs.traces(:,~newBadROIs, :);
    
    
    %%%%%%%% Set traces_al_stim: same as traces_al_stimAll except that some trials are
    % set to nan: bc their stim duration is < th_stim_dur ms or bc their go
    % tone happens before ep(end) ms. (In traces_al_stimAll, all trials are
    % included).
    traces_al_stimAll = stimAl_allTrs.traces;
    time_aligned_stim = stimAl_allTrs.time;
    eventI = stimAl_allTrs.eventI;
    
    popClassifier_setToRmv % set toRmv, ie trials in which go tone happened before ep_ms(end) and trials that have <th_stim_dur duration.
    
    traces_al_stim = traces_al_stimAll;
    traces_al_stim(:,:,toRmv) = nan;
    
    
    % Set X, Y, TrsExcluded, NsExcluded
    popClassifier_setXY % set X, Y, TrsExcluded, NsExcluded
    
    % Exclude nan trials and non-active neurons from traces_al_stim
    traces_al_stim = traces_al_stim(:, ~NsExcluded, ~trsExcluded);
    
    % Exclude nan trials from stimrate
    stimrate = stimrate(~trsExcluded);
    
    
    %% 
    
    inhRois = inhibitRois(~NsExcluded);
    
    % traces_al_stim = traces_al_stim(:,inhRois==nt,:);
    fe = find(inhRois==nt); % exc or inh analysis?
    
    T = size(traces_al_stim,1);
    regressBins = 2;
    t = floor(T/regressBins);
    if nt==0
        rmax = 10;
    elseif nt==1
        rmax = 1;
    end
    
    angle_all = nan(t,t,4,rmax);
    aIx_all = nan(t,t,rmax);
    traces_al_stim_orig = traces_al_stim;
    X_orig = X;
    
    
    for r = 1:rmax % for exc, on each round randomly select n neurons, n=number of inh
        
        rr = randperm(length(fe));
        fnow = fe(rr(1:sum(inhRois==1)));
        
        traces_al_stim = traces_al_stim_orig(:,fnow,:);
        X = X_orig(:,fnow);
        % size(X), size(traces_al_stim)
        
        
        %%
        if nt==2 % for now, needs work for nt~=2
            % Then I identify a random subset of neurons ( = 0.95 * numTrials):
            % ##  If number of neurons is more than 95% of trial numbers, identify n random neurons, where n= 0.95 * number of trials. This is to make sure we have more observations (trials) than features (neurons)
            
            nTrs = size(X, 1);
            nNeuronsOrig = size(X, 2);
            nNeuronsNow = floor(nTrs * .95);
            
            if nNeuronsNow < nNeuronsOrig
                fractInh = sum(inhRois==1) / (nNeuronsOrig);
                fractExc = sum(inhRois==0) / (nNeuronsOrig);
                fractUn = sum(isnan(inhRois)) / (nNeuronsOrig);
                sprintf( 'Number: inhibit = %d, excit = %d, unsure = %d', sum(inhRois==1), sum(inhRois==0), sum(isnan(inhRois)))
                sprintf(' Fraction: inhibit = %.2f, excit = %.2f, unsure = %.2f', fractInh, fractExc, fractUn)
                % Define how many neurons you need to pick from each pool of inh, exc, unsure.
                nInh = ceil(fractInh*nNeuronsNow);
                nExc = ceil(fractExc*nNeuronsNow);
                nUn = nNeuronsNow - (nInh + nExc); % fractUn*nNeuronsNow
                
                sprintf ('\nThere are %d trials; So selecting %d neurons out of %d ', nTrs,nNeuronsNow, nNeuronsOrig)
                sprintf('%i, %i, %i: number of selected inh, excit, unsure' ,nInh, nExc, nUn)
                
                % Select nInh random indeces out of the inhibitory pool
                inhI = find(inhRois==1);
                inhNow = inhI(randperm(length(inhI)));inhNow = sort(inhNow(1:nInh)); % # random indices
                
                
                % Select nExc random indeces out of the excitatory pool
                excI = find(inhRois==0);
                excNow = excI(randperm(length(excI)));excNow = sort(excNow(1:nExc)); % # random indices
                
                
                % Select nUn random indeces out of the unsure pool
                unI = find(isnan(inhRois));
                unNow = unI(randperm(length(unI)));unNow = sort(unNow(1:nUn)); % # random indices
                
                % Put all the 3 groups together
                neuronsNow = sort([inhNow(:); excNow(:); unNow(:)]);
                
                
                % Define a logical array with 1s for randomly selected neurons (length = number of neurons in X (after excluding NsExcluded))
                NsRand = 1:size(X,2);
                NsRand = ismember(NsRand, neuronsNow);
                
            else % if number of neurons is already <= .95*numTrials, include all neurons.
                sprintf('Not doing random selection of neurons (nNeurons=%d already fewer than .95*nTrials=%d)', size(X,2), nTrs)
                NsRand = true(size(X,2),1);
            end
            
            
            
            % Finally I extract only NsRand from X and inhRois:
            % Set X and inhRois only for the randomly selected set of neurons
            
            X = X(:,NsRand);
            inhRois = inhRois(NsRand);
            traces_al_stim = traces_al_stim(:, NsRand, :);
        end
        
        
        %% Stability analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%
        dataTensor = traces_al_stim; % non_filtered; % stimulus aligned traces. Only for active neurons and valid (non-nan) trials.
        nan_msk = ~squeeze(isnan(sum(sum(dataTensor,1),2)));
        dataTensor = dataTensor(:, :, nan_msk);
        all_times = time_aligned_stim; % time_aligned;
        % dataTensor = traces_al_1stSideTry;
        % all_times = time_aligned_1stSideTry;
        
        [T, N, R] = size(dataTensor);
        if R<N
            error(' number of trials < number of neurons: results may be inaccurate');
        end
        
        
        %% average across multiple times (downsample dataTensor; not a moving average. we only average every regressBins points.)
        % regressBins = 2;
        dataTensor = dataTensor(1:regressBins*floor(T/regressBins), : , :);
        dataTensor = squeeze(mean(reshape(dataTensor, regressBins, floor(T/regressBins), N, R), 1)); % not a moving average. we only average every regressBins points.
        all_times = all_times(1:floor(T/regressBins)*regressBins);
        all_times = round(mean(reshape(all_times, regressBins, floor(T/regressBins)),1), 2);
        
        
        %% preprocess dataTensor: do mean subtraction and feature normalization (for each neuron)
        [T, N, R] = size(dataTensor);
        meanN = mean(reshape(permute(dataTensor, [1 3 2]), T*R, N)); % 1xneurons; includes the average of all time points for all trials for each neuron.
        stdN = std(reshape(permute(dataTensor, [1 3 2]), T*R, N));
        meanN = mean(X); % X: trials x neurons; includes average of window ep (defined in SVM codes) for each neuron at each trial.
        stdN = std(X);
        dataTensor = bsxfun(@times, bsxfun(@minus, dataTensor, meanN), 1./(stdN+sqrt(0)));
        smooth_dataTensor = nan(size(dataTensor));
        for i = 1: size(dataTensor,3)
            for j = 1:size(dataTensor,2)
                smooth_dataTensor(:, j , i) = gaussFilter(2, dataTensor(:,j,i));
            end
        end
        
        
        %%% plot average of dataTensor per decision (average across trials and neurons)
        if doplots
            figure;
            hold on
            plot(all_times, mean(mean(dataTensor(:, :, Y==0), 3), 2), 'b')
            plot(all_times, mean(mean(dataTensor(:, :, Y==1), 3), 2), 'r')
            plot(all_times, mean(mean(dataTensor(:, :, :), 3), 2), 'k')
            legend('average low rate', 'average high rate', 'average all')
            xlabel('time (ms)') % since stimulus onset.
            ylabel('normalized firing rates')
        end
        
        
        %% Alignment of top variance subspaces
        
        numDim = 10; % define dimensionality of subspace
        [PCs_t, Summary] = pca_t(dataTensor, numDim); % identify subspaces (FN: each row is the PC space (neurons x numPCs) for a particular time point).
        aIx = nan(T, T); % alignement index between subspaces
        for i = 1:T
            for j = 1:T
                aIx(i,j) = alignIx(squeeze(PCs_t(i, :, :)), squeeze(PCs_t(j, :, :))); % compute alignment between PC spaces at time points i and j. % FN: this is done by computing variance explained after projecting pc_i onto pc_j divided by variance explained before the projection.
            end
        end
        aIx_all(:,:,r) = aIx;
        
        
        if doplots
            figure;
            imagesc(all_times, all_times, aIx);
            colormap(colormap('jet'))
            title(['alignment index (' num2str(numDim) ' PCs)'])
            colorbar
            caxis([0 1])
            axis square
            xlabel('time (ms)')
            ylabel('time (ms)')
        end
        
        
        %% TDR analysis: FN: 1st we do denoising. Then for each neuron and at each timepoint, we fit a regression model that predicts neural response given stimulus and choice.
        
        stim = stimrate(:);
        decision = Y;
        cb = 16;
        kfold = 10;
        codedParams = [[stim(:)-cb]/range(stim(:)) [decision(:)-mean(decision(:))]/range(decision(:)) ones(R, 1)]; % trials x 1: stimulusRate_choice_ones; stimulusRate and choice are mean subtracted.
        [~, codedParams] = normVects(codedParams); % each parameter is unit vector % normalize each column to have length 1.
        % cb = 16; stimrate_norm = ((stimrate - cb)/max(abs(stimrate(:) - cb)));
        numPCs = 10;
        [dRAs, normdRAs, Summary] = runTDR(dataTensor, numPCs, codedParams, [], kfold);
        % dRAs: times(frames) x neurons x predictors; normalized beta coefficients (ie coefficients of stimulus, choice, and offset for each neuron at each timepoint)
        
        
        % Now we compute the angle (times x times) between 2 subspaces of neurons
        % that represent particular features (stimulus,choice,offset) at all timepoints.
        %
        % angle(t1,t2,4) = Angle between 2 vectors of size 1 x neurons. 1 vector
        % represents stimulus beta at time t1 (dRAs(t1,:,1)) and the other
        % represents choice beta at time t2 dRAs(t2,:,1).
        angle = real(acos(abs(dRAs(:,:,1)*dRAs(:,:,1)')))*180/pi; % times x times
        angle(:,:,2) = real(acos(abs(dRAs(:,:,2)*dRAs(:,:,2)')))*180/pi;
        angle(:,:,3) = real(acos(abs(dRAs(:,:,3)*dRAs(:,:,3)')))*180/pi;
        angle(:,:,4) = real(acos(abs(dRAs(:,:,1)*dRAs(:,:,2)')))*180/pi;
        
        %%%
        angle_all(:,:,:,r) = angle;        
        
    end
    
    
    if nt==0
        aIx_all_alld_exc{iday} = aIx_all;
        angle_all_alld_exc{iday} = angle_all;
    elseif nt==1
        aIx_all_alld_inh{iday} = aIx_all;
        angle_all_alld_inh{iday} = angle_all;
    end
    
    
end


%%
if trialHistAnalysis==0
    if exist('stab_all_curr.mat', 'file')==2
        if nt==0
            save('stab_all_curr', '-append', 'aIx_all_alld_exc', 'angle_all_alld_exc')
        elseif nt==1
            save('stab_all_curr', '-append', 'aIx_all_alld_inh', 'angle_all_alld_inh')
        end
    else
        if nt==0
            save('stab_all_curr', 'aIx_all_alld_exc', 'angle_all_alld_exc')
        elseif nt==1
            save('stab_all_curr', 'aIx_all_alld_inh', 'angle_all_alld_inh')
        end
    end
else
    save stab_all_prev aIx_all_alld_exc aIx_all_alld_inh angle_all_alld_exc angle_all_alld_inh
end



%% Compute eventI for all days 

set_eventI_allDays;

 
%% Go with min nPre and nPost across days

% Find the common eventI, number of frames before and after the common eventI for the alignment of traces of all days.
% By common eventI, we  mean the index on which all traces will be aligned.

nPost = nan(1,length(days));
for iday = 1:length(days)
    nPost(iday) = size(aIx_all_alld_exc{iday},1) - eventI_allDays(iday);
end

nPreMin = min(eventI_allDays)-1;
nPostMin = min(nPost);
fprintf('Number of frames before = %d, and after = %d the common eventI', nPreMin, nPostMin)


%%% Set the time array for the across-day aligned traces
a = -frameLength * (0:nPreMin); a = a(end:-1:1);
b = frameLength * (1:nPostMin);
time_aligned = [a,b];
size(time_aligned)


%%% Align traces of all days on the common eventI

aIx_all_alld_exc_aligned = cell(1,length(days)); % nan(nPreMin + nPostMin + 1, length(days));
aIx_all_alld_inh_aligned = cell(1,length(days)); 
angle_all_alld_exc_aligned = cell(1,length(days));
angle_all_alld_inh_aligned = cell(1,length(days));

for iday = 1:length(days)
    r = eventI_allDays(iday) - nPreMin  :  eventI_allDays(iday) + nPostMin;
    aIx_all_alld_exc_aligned{iday} = aIx_all_alld_exc{iday}(r,r, :);
    aIx_all_alld_inh_aligned{iday} = aIx_all_alld_inh{iday}(r,r, :);
    
    angle_all_alld_exc_aligned{iday} = angle_all_alld_exc{iday}(r,r,:, :);
    angle_all_alld_inh_aligned{iday} = angle_all_alld_inh{iday}(r,r,:, :);
end



%% Go with max nPre and nPost and use nans

nPreMax = max(eventI_allDays)-1;
nPostMax = max(nPost);
len = nPreMax + nPostMax + 1;

%%% Set the time array for the across-day aligned traces
a = -frameLength * (0:nPreMax); a = a(end:-1:1);
b = frameLength * (1:nPostMax);
time_aligned = [a,b];
size(time_aligned)


%%% Align traces of all days on the common eventI
aIx_all_alld_exc_aligned = cell(1,length(days)); % 
aIx_all_alld_inh_aligned = cell(1,length(days)); 
angle_all_alld_exc_aligned = cell(1,length(days));
angle_all_alld_inh_aligned = cell(1,length(days));

for iday = 1:length(days)
    r = eventI_allDays(iday) - nPreMin  :  eventI_allDays(iday) + nPostMin;
    
    aIx_all_alld_exc_aligned{iday} = nan(len, len, size(aIx_all_alld_exc{iday},3)); % aIx_all_alld_exc{iday}(r,r, :);
    aIx_all_alld_inh_aligned{iday} = nan(len, len, size(aIx_all_alld_inh{iday},3));    

    angle_all_alld_exc_aligned{iday} = nan(len, len, size(angle_all_alld_exc{1},3), size(aIx_all_alld_exc{iday},3)); % aIx_all_alld_exc{iday}(r,r, :);
    angle_all_alld_inh_aligned{iday} = nan(len, len, size(angle_all_alld_exc{1},3), size(aIx_all_alld_inh{iday},3));    

end


%%%%
for iday = 1:length(days)
    beg = nPreMax - (eventI_allDays(iday)-1)+1;
    lenNow = size(aIx_all_alld_exc{iday},1); 
    r = beg :  beg+lenNow-1;
    
    aIx_all_alld_exc_aligned{iday}(r,r,:) = aIx_all_alld_exc{iday};
    aIx_all_alld_inh_aligned{iday}(r,r,:) = aIx_all_alld_inh{iday};
    
    angle_all_alld_exc_aligned{iday}(r,r,:,:) = angle_all_alld_exc{iday};
    angle_all_alld_inh_aligned{iday}(r,r,:,:) = angle_all_alld_inh{iday};
end



%% Pool results of all days

% average across rounds
aIxexc = cellfun(@(x)nanmean(x,3), aIx_all_alld_exc_aligned, 'uniformoutput',0); % average across rounds for each day
aIxinh = cellfun(@(x)nanmean(x,3), aIx_all_alld_inh_aligned, 'uniformoutput',0); % average across rounds for each day

angexc = cellfun(@(x)nanmean(x,4), angle_all_alld_exc_aligned, 'uniformoutput',0); % average across rounds for each day
anginh = cellfun(@(x)nanmean(x,4), angle_all_alld_inh_aligned, 'uniformoutput',0); % average across rounds for each day


% Average across days
a = reshape(cell2mat(aIxexc), size(aIxexc{1},1), size(aIxexc{1},1), []);
aIxExcAve = nanmean(a, 3);
b = reshape(cell2mat(aIxinh), size(aIxexc{1},1), size(aIxexc{1},1), []);
aIxInhAve = nanmean(b, 3);


a = reshape(cell2mat(angexc), size(aIxexc{1},1), size(aIxexc{1},1), size(angle_all_alld_exc{1},3), []);
angExcAve = nanmean(a, 4);
b = reshape(cell2mat(anginh), size(aIxexc{1},1), size(aIxexc{1},1), size(angle_all_alld_exc{1},3), []);
angInhAve = nanmean(b, 4);



%%
figure; 
subplot(221),imagesc(aIxExcAve), colormap jet, title('exc')
subplot(222), imagesc(aIxInhAve), colormap jet, title('inh')


%%
angle = angExcAve;
% angle = angInhAve;

figure('name', 'exc angle');
subplot(221)
imagesc(all_times, all_times, angle(:,:,1));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(222)
imagesc(all_times, all_times, angle(:,:,2));
colormap(flipud(colormap('jet')))
colorbar
title('angle decision')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(223)
imagesc(all_times, all_times, angle(:,:,3));
colormap(flipud(colormap('jet')))
colorbar
title('angle offset')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(224)
imagesc(all_times, all_times, angle(:,:,4));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim vs decision')
caxis([0 90])
xlabel('decision time (ms)')
ylabel('stimulus time (ms)')
axis square



angle = angInhAve;

figure('name', 'inh angle');
subplot(221)
imagesc(all_times, all_times, angle(:,:,1));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(222)
imagesc(all_times, all_times, angle(:,:,2));
colormap(flipud(colormap('jet')))
colorbar
title('angle decision')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(223)
imagesc(all_times, all_times, angle(:,:,3));
colormap(flipud(colormap('jet')))
colorbar
title('angle offset')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(224)
imagesc(all_times, all_times, angle(:,:,4));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim vs decision')
caxis([0 90])
xlabel('decision time (ms)')
ylabel('stimulus time (ms)')
axis square


%% Look at each day one by one

f1 = figure;
f2 = figure;
f3 = figure;

for i = 1:length(days)
    
    figure(f1)
    subplot(221)
    imagesc(time_aligned, time_aligned, nanmean(aIx_all_alld_exc{i},3));
    colormap('jet')
    colorbar
    caxis([0 1])
    axis square
    xlabel('time (ms)')
    ylabel('time (ms)')
    
    subplot(222)
    imagesc(time_aligned, time_aligned, nanmean(aIx_all_alld_inh{i},3));
    colormap('jet')
%     title(['alignment index (' num2str(numDim) ' PCs)'])
    colorbar
    caxis([0 1])
    axis square
    xlabel('time (ms)')
    ylabel('time (ms)')

    
    %%
    figure(f2)
    angle = angexc{i};
    
    subplot(221)
    imagesc(all_times, all_times, angle(:,:,1));
    colormap(flipud(colormap('jet')))
    colorbar
    title('angle stim')
    caxis([0 90])
    xlabel('time (ms)')
    ylabel('time (ms)')
    axis square

    subplot(222)
    imagesc(all_times, all_times, angle(:,:,2));
    colormap(flipud(colormap('jet')))
    colorbar
    title('angle decision')
    caxis([0 90])
    xlabel('time (ms)')
    ylabel('time (ms)')
    axis square

    subplot(223)
    imagesc(all_times, all_times, angle(:,:,3));
    colormap(flipud(colormap('jet')))
    colorbar
    title('angle offset')
    caxis([0 90])
    xlabel('time (ms)')
    ylabel('time (ms)')
    axis square

    subplot(224)
    imagesc(all_times, all_times, angle(:,:,4));
    colormap(flipud(colormap('jet')))
    colorbar
    title('angle stim vs decision')
    caxis([0 90])
    xlabel('decision time (ms)')
    ylabel('stimulus time (ms)')
    axis square


    
    
    figure(f3)
    angle = anginh{i};
    
    subplot(221)
    imagesc(all_times, all_times, angle(:,:,1));
    colormap(flipud(colormap('jet')))
    colorbar
    title('angle stim')
    caxis([0 90])
    xlabel('time (ms)')
    ylabel('time (ms)')
    axis square

    subplot(222)
    imagesc(all_times, all_times, angle(:,:,2));
    colormap(flipud(colormap('jet')))
    colorbar
    title('angle decision')
    caxis([0 90])
    xlabel('time (ms)')
    ylabel('time (ms)')
    axis square

    subplot(223)
    imagesc(all_times, all_times, angle(:,:,3));
    colormap(flipud(colormap('jet')))
    colorbar
    title('angle offset')
    caxis([0 90])
    xlabel('time (ms)')
    ylabel('time (ms)')
    axis square

    subplot(224)
    imagesc(all_times, all_times, angle(:,:,4));
    colormap(flipud(colormap('jet')))
    colorbar
    title('angle stim vs decision')
    caxis([0 90])
    xlabel('decision time (ms)')
    ylabel('stimulus time (ms)')
    axis square

    
    pause
end


%%
error('stop')

%%
figure;
imagesc(all_times, all_times, nanmean(aIx_all,3));
colormap(colormap('jet'))
title(['alignment index (' num2str(numDim) ' PCs)'])
colorbar
caxis([0 1])
axis square
xlabel('time (ms)')
ylabel('time (ms)')

%%
angle = nanmean(angle_all,4);
% angle = angExcAve;
% angle = angInhAve;

figure;
subplot(221)
imagesc(all_times, all_times, angle(:,:,1));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(222)
imagesc(all_times, all_times, angle(:,:,2));
colormap(flipud(colormap('jet')))
colorbar
title('angle decision')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(223)
imagesc(all_times, all_times, angle(:,:,3));
colormap(flipud(colormap('jet')))
colorbar
title('angle offset')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(224)
imagesc(all_times, all_times, angle(:,:,4));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim vs decision')
caxis([0 90])
xlabel('decision time (ms)')
ylabel('stimulus time (ms)')
axis square

%%
error('stop')

%%
figure;
subplot(221)
plot(all_times, Summary.R2_t, 'k')
xlabel('time (ms)')
ylabel('R^2')

subplot(222);
hold on
plot(all_times, Summary.R2_tk(:,1), 'g')
plot(all_times, Summary.R2_tk(:,2), 'b')
plot(all_times, Summary.R2_tk(:,3), 'c')
xlabel('time (ms)')
ylabel('R2 of each predictor') % signal contribution to firing rate
legend('stimulus', 'decision', 'offset')

subplot(223); hold on;
a = mean(dRAs(:,:,1),2);
plot(all_times, a, 'g')
a = mean(dRAs(:,:,2),2);
plot(all_times, a, 'b')
a = mean(dRAs(:,:,3),2);
plot(all_times, a, 'c')
ylabel({'Normalized beta', '(average across neurons)'})
legend('stimulus', 'decision', 'offset')



figure;
subplot(221)
imagesc(all_times, all_times, angle(:,:,1));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(222)
imagesc(all_times, all_times, angle(:,:,2));
colormap(flipud(colormap('jet')))
colorbar
title('angle decision')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(223)
imagesc(all_times, all_times, angle(:,:,3));
colormap(flipud(colormap('jet')))
colorbar
title('angle offset')
caxis([0 90])
xlabel('time (ms)')
ylabel('time (ms)')
axis square

subplot(224)
imagesc(all_times, all_times, angle(:,:,4));
colormap(flipud(colormap('jet')))
colorbar
title('angle stim vs decision')
caxis([0 90])
xlabel('decision time (ms)')
ylabel('stimulus time (ms)')
axis square

%% non-orthogonal projections
stim_epoch = all_times>=30 & all_times<=floor(min(timeStimOffset-timeStimOnset));
dec_epoch = all_times>=(floor(min(time1stSideTry-timeStimOnset))-330) & all_times<=(floor(min(time1stSideTry-timeStimOnset))-30);
[~, sRA_star] = normVects([mean(dRAs(stim_epoch ,:,1)).' mean(dRAs(all_times>0,:,2)).']);

dataTensor_proj(:, 1, :) = projectTensor(smooth_dataTensor, squeeze(sRA_star(:, 1)));
dataTensor_proj(:, 2, :) = projectTensor(smooth_dataTensor, squeeze(sRA_star(:, 2)));

uniqueStim = unique(stim);
uniqueDecision = unique(decision);
S = length(uniqueStim);
D = length(uniqueDecision);
projStim = [];
for s = 1:S
    for d = 1:D
        msk = (stim == uniqueStim(s)) & (decision == uniqueDecision(d));
        proj1(:, s, d) = mean(squeeze(dataTensor_proj(: ,1, msk)), 2);
        proj2(:, s, d) = mean(squeeze(dataTensor_proj(: ,2, msk)), 2);
    end
end

clr = redgreencmap(S, 'interpolation', 'linear');
figure;
l = {};
subplot(121)
hold on
for s = 1:S
    plot(all_times, proj1(:, s, 1), '--', 'color', clr(s, :));
    h(s) = plot(all_times, proj1(:, s, 2), '-', 'color', clr(s, :));
    l{s} = [num2str(uniqueStim(s)) 'Hz'];
end
ylabel('stimulus projection')
legend(h, l)
xlabel('time (ms)') % since stimulus onset.


subplot(122)
hold on
for s = 1:S
    plot(all_times, proj2(:, s, 1), '--', 'color', clr(s, :));
    h(s) = plot(all_times, proj2(:, s, 2), '-', 'color', clr(s, :));
end
ylabel('decsison projection')
legend(h, l)
xlabel('time (ms)') % since stimulus onset.


mean_stim = mean(squeeze(dataTensor_proj(:,1,:)),2);
mean_dec = mean(squeeze(dataTensor_proj(:,2,:)),2);

var_stim = var(squeeze(dataTensor_proj(:,1,:)),[], 2);
var_dec = var(squeeze(dataTensor_proj(:,2,:)),[], 2);
figure
subplot(121)
plot(mean_stim, mean_dec)
hold on
plot(mean_stim(find(all_times>=0, 1, 'first')), mean_dec(find(all_times>=0, 1, 'first')), 'ro')
xlabel('mean activity (stim dim)')
ylabel('mean activity (decision dim)')
axis square

subplot(122)
plot(var_stim, var_dec)
hold on
plot(var_stim(find(all_times>=0, 1, 'first')), var_dec(find(all_times>=0, 1, 'first')), 'ro')
xlabel('var. activity (stim dim)')
ylabel('var. activity (decision dim)')
axis square
%%
% % % % % % %% orthogonal projections
% % % % % % [sRA, g] = optimize_oTDR(dataTensor, codedParams, [], []);
% % % % % %
% % % % % % figure;
% % % % % % hold on
% % % % % % plot(all_times, g(:,1), 'r')
% % % % % % plot(all_times, g(:,2), 'b')
% % % % % % hold off
% % % % % % xlabel('time (ms)')
% % % % % % ylabel('magnitude')
% % % % % % legend('stimulus', 'decision')
% % % % % %
% % % % % % dataTensor_proj(:, 1, :) = projectTensor(dataTensor, squeeze(sRA(:, 1)));
% % % % % % dataTensor_proj(:, 2, :) = projectTensor(dataTensor, squeeze(sRA(:, 2)));
% % % % % %
% % % % % % uniqueStim = unique(stim);
% % % % % % uniqueDecision = unique(decision);
% % % % % % S = length(uniqueStim);
% % % % % % D = length(uniqueDecision);
% % % % % % projStim = [];
% % % % % % for s = 1:S
% % % % % %     for d = 1:D
% % % % % %         msk = (stim == uniqueStim(s)) & (decision == uniqueDecision(d));
% % % % % %         proj1(:, s, d) = mean(squeeze(dataTensor_proj(: ,1, msk)), 2);
% % % % % %         proj2(:, s, d) = mean(squeeze(dataTensor_proj(: ,2, msk)), 2);
% % % % % %     end
% % % % % % end
% % % % % %
% % % % % % clr = redgreencmap(S, 'interpolation', 'linear');
% % % % % % figure;
% % % % % % l = {};
% % % % % % subplot(121)
% % % % % % hold on
% % % % % % for s = 1:S
% % % % % %     plot(all_times, proj1(:, s, 1), '--', 'color', clr(s, :));
% % % % % %     h(s) = plot(all_times, proj1(:, s, 2), '-', 'color', clr(s, :));
% % % % % %     l{s} = [num2str(uniqueStim(s)) 'Hz'];
% % % % % % end
% % % % % % ylabel('stimulus projection')
% % % % % % legend(h, l)
% % % % % % xlabel('time (ms)') % since stimulus onset.
% % % % % %
% % % % % %
% % % % % % subplot(122)
% % % % % % hold on
% % % % % % for s = 1:S
% % % % % %     plot(all_times, proj2(:, s, 1), '--', 'color', clr(s, :));
% % % % % %     h(s) = plot(all_times, proj2(:, s, 2), '-', 'color', clr(s, :));
% % % % % % end
% % % % % % ylabel('decsison projection')
% % % % % % legend(h, l)
% % % % % % xlabel('time (ms)') % since stimulus onset.
% % % % % %
% % % % % %
% % % % % % figure
% % % % % % subplot(121)
% % % % % % plot(mean(squeeze(dataTensor_proj(:,1,:))), mean(squeeze(dataTensor_proj(:,2,:))))
% % % % % % xlabel('mean activity (stim dim)')
% % % % % % ylabel('mean activity (decision dim)')
% % % % % % axis square
% % % % % %
% % % % % % subplot(122)
% % % % % % plot(var(squeeze(dataTensor_proj(:,1,:))), var(squeeze(dataTensor_proj(:,2,:))))
% % % % % % xlabel('mean activity (stim dim)')
% % % % % % ylabel('mean activity (decision dim)')
% % % % % % axis square
%%
saving_directory = ['/Users/gamalamin/git_local_repository/Farzaneh/results/' imagingFolder '/'];
saveFig2Directory(saving_directory)
save([saving_directory 'allresults.mat'])
toc