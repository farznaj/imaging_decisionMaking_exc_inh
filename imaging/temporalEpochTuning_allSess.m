% Plot for each frame in the trial, the fraction of neurons that have their
% peak activity in that given frame... this is our measure of fraction of
% neurons that are tuned to each epoch (time point) in the trial.

% vars are set and saved in avetrialAlign_plotAve_trGroup. Here we load
% them, align them across sessions, and average them across sessions.


%%
savefigs = 1;
doSingleDayPlots = 0;

mice = {'fni16','fni17','fni18','fni19'};
dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/TemporalEpochTuning';


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
    
    
    %% Set vars to align traces across sessions
    
    npre_allSess = nan(length(days), 4);
    npost_allSess = nan(length(days), 4);
    % fractNs_maxInEachFr_exc_allsess = cell(1, length(days));
    % fractNs_maxInEachFr_inh_allsess = cell(1, length(days));
    
    for iday = 1:length(days)
        
        disp('__________________________________________________________________')
        dn = simpleTokenize(days{iday}, '_');
        
        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
        
        fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})
        
        
        %% Set vars
        
        signalCh = 2; % because you get A from channel 2, I think this should be always 2.
        pnev2load = [];
        [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
        % [pd, pnev_n] = fileparts(pnevFileName);
        [md,date_major] = fileparts(imfilename);
        cd(md) % r = repmat('%03d-', 1, length(mdfFileNumber)); r(end) = []; date_major = sprintf(['%s_', r], imagingFolder, mdfFileNumber);
        
        
        %% Load temp tuning vars
        
        clear fractNs_maxInEachFr_exc', 'fractNs_maxInEachFr_inh
        
        [pd, pnev_n] = fileparts(pnevFileName);
        postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
        
        [~,b] = fileparts(postName);
        fractName = ['tempTuning_', b(6:end)];
        
        load(fractName, 'fractNs_maxInEachFr_exc', 'fractNs_maxInEachFr_inh',...
            'eventI_initTone', 'eventI_stimOn', 'eventI_1stSideTry', 'eventI_reward',...
            'lenTrace_initTone', 'lenTrace_stimOn', 'lenTrace_1stSideTry', 'lenTrace_reward')
        
        %% Keep vars of all sessions
        
        %     fractNs_maxInEachFr_exc_allsess{iday} = fractNs_maxInEachFr_exc;
        %     fractNs_maxInEachFr_inh_allsess{iday} = fractNs_maxInEachFr_inh;
        
        %% Set npre , npost
        
        npre_init = eventI_initTone - 1;
        npre_stim = eventI_stimOn - 1;
        npre_choice = eventI_1stSideTry - 1;
        npre_rew = eventI_reward - 1;
        
        npost_init = lenTrace_initTone - eventI_initTone;
        npost_stim = lenTrace_stimOn - eventI_stimOn;
        npost_choice = lenTrace_1stSideTry - eventI_1stSideTry;
        npost_rew = lenTrace_reward - eventI_reward;
        
        
        npre_all = [npre_init, npre_stim, npre_choice, npre_rew];
        npost_all = [npost_init, npost_stim, npost_choice, npost_rew];
        % [lenTrace_initTone, lenTrace_stimOn, lenTrace_1stSideTry, lenTrace_reward]
        
        npre_allSess(iday,:) = npre_all;
        npost_allSess(iday,:) = npost_all;
        
    end
    
    [(1:length(days))', npre_allSess]
    [(1:length(days))', npost_allSess]
    
    
    %% Set npre and npost across all sessions
    
    % because the traces in the first days are very short, we dont use them for
    % alignment; we start with the day that has at least 5 frames (*regressBin)
    % before the stimulus onset:
    
    if mouse=='fni18'
        days2use = 1 : length(days);
    else
        find(npost_allSess(:,2)==5, 1)
        days2use = find(npost_allSess(:,2)==5, 1) : length(days);
    end
    
    npre_min = min(npre_allSess(days2use,:))
    npost_min = min(npost_allSess(days2use,:))
    
    
    %% Align
    
    fractNs_maxInEachFr_exc_aligned = nan(length(days2use), sum(npre_min+1) + sum(npost_min));
    fractNs_maxInEachFr_inh_aligned = nan(length(days2use), sum(npre_min+1) + sum(npost_min));
    cnt = 0;
    
    for iday = days2use %1:length(days)
        cnt = cnt+1;
        disp('__________________________________________________________________')
        dn = simpleTokenize(days{iday}, '_');
        
        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
        
        fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})
        
        
        %% Set vars
        
        signalCh = 2; % because you get A from channel 2, I think this should be always 2.
        pnev2load = [];
        [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
        % [pd, pnev_n] = fileparts(pnevFileName);
        [md,date_major] = fileparts(imfilename);
        cd(md) % r = repmat('%03d-', 1, length(mdfFileNumber)); r(end) = []; date_major = sprintf(['%s_', r], imagingFolder, mdfFileNumber);
        
        
        %% Load temp tuning vars
        
        clear fractNs_maxInEachFr_exc', 'fractNs_maxInEachFr_inh
        
        [pd, pnev_n] = fileparts(pnevFileName);
        postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
        
        [~,b] = fileparts(postName);
        fractName = ['tempTuning_', b(6:end)];
        
        load(fractName, 'fractNs_maxInEachFr_exc', 'fractNs_maxInEachFr_inh',...
            'eventI_initTone', 'eventI_stimOn', 'eventI_1stSideTry', 'eventI_reward',...
            'lenTrace_initTone', 'lenTrace_stimOn', 'lenTrace_1stSideTry', 'lenTrace_reward')
        
        %% Align
        
        fractNs_maxInEachFr_exc_aligned(cnt,:) = [fractNs_maxInEachFr_exc(eventI_initTone - npre_min(1) : eventI_initTone + npost_min(1)), ...
            fractNs_maxInEachFr_exc(lenTrace_initTone + eventI_stimOn - npre_min(2) : lenTrace_initTone + eventI_stimOn + npost_min(2)), ...
            fractNs_maxInEachFr_exc(lenTrace_initTone + lenTrace_stimOn + eventI_1stSideTry - npre_min(3) : lenTrace_initTone + lenTrace_stimOn + eventI_1stSideTry + npost_min(3)), ...
            fractNs_maxInEachFr_exc(lenTrace_initTone + lenTrace_stimOn + lenTrace_1stSideTry + eventI_reward - npre_min(4) : lenTrace_initTone + lenTrace_stimOn + lenTrace_1stSideTry + eventI_reward + npost_min(4))];
        
        fractNs_maxInEachFr_inh_aligned(cnt,:) = [fractNs_maxInEachFr_inh(eventI_initTone - npre_min(1) : eventI_initTone + npost_min(1)), ...
            fractNs_maxInEachFr_inh(lenTrace_initTone + eventI_stimOn - npre_min(2) : lenTrace_initTone + eventI_stimOn + npost_min(2)), ...
            fractNs_maxInEachFr_inh(lenTrace_initTone + lenTrace_stimOn + eventI_1stSideTry - npre_min(3) : lenTrace_initTone + lenTrace_stimOn + eventI_1stSideTry + npost_min(3)), ...
            fractNs_maxInEachFr_inh(lenTrace_initTone + lenTrace_stimOn + lenTrace_1stSideTry + eventI_reward - npre_min(4) : lenTrace_initTone + lenTrace_stimOn + lenTrace_1stSideTry + eventI_reward + npost_min(4))];
        
        
    end
    
    
    % check
    if ~isequal(size(fractNs_maxInEachFr_exc_aligned,2) , size(fractNs_maxInEachFr_inh_aligned,2) , sum(npre_min+1) + sum(npost_min))
        error('something wrong with the alignment')
    end
    
    
    %% Plot aligned traces (averaged across sessions)
    
    me = nanmean(fractNs_maxInEachFr_exc_aligned, 1);
    se = nanstd(fractNs_maxInEachFr_exc_aligned, [], 1) / sqrt(length(days2use));
    mi = nanmean(fractNs_maxInEachFr_inh_aligned, 1);
    si = nanstd(fractNs_maxInEachFr_inh_aligned, [], 1) / sqrt(length(days2use));
    
    [h,p] = ttest2(fractNs_maxInEachFr_exc_aligned', fractNs_maxInEachFr_inh_aligned');
    pp = nan(1, length(p));
    pp(p<.05) = 1;
    
    figure('name', [mouse, ' - ave days'], 'position', [680   453   358   516]);
    subplot(211), hold on
    plot(me, 'b')
    plot(mi, 'r') % plot(smooth(fractNs_maxInEachFr_inh, 3), 'g')
    vline(cumsum(npre_min + npost_min + 1) - npost_min, 'k:')
    yl = get(gca, 'ylim');
    plot(pp * yl(2)+.005, 'g-')
    legend('exc', 'inh')
    xlabel('Frame (downsampled)')
    ylabel('Fract Ns with Max in each frame')
    title(['init , stim , choice , reward'])
    
    subplot(212), hold on
    boundedline(1:size(fractNs_maxInEachFr_exc_aligned,2), me+se, me-se, 'b', 'alpha')
    boundedline(1:size(fractNs_maxInEachFr_exc_aligned,2), mi+si, mi-si, 'r', 'alpha')
    vline(cumsum(npre_min + npost_min + 1) - npost_min, 'k:')
    
    if savefigs
        fign = strcat(mouse, '_aveDays_fractNs_maxInEachFr_excInh');
        savefig(gcf, fullfile(dirn0, fign))
        print(gcf, '-dpdf', fullfile(dirn0, fign))
    end
    
    
    
    
    %% Plot temporal epoch tuning for each day
    
    if doSingleDayPlots
        fign = strcat(mouse, '_fractNs_maxInEachFr_excInh');
        
        r = ceil(length(days)/10);
        c = ceil(length(days)/r);
        
        f = figure('name', mouse);
        ha = tight_subplot(r,c,[.05,.03],.05,.05);
        
        %%%
        for iday = 1:length(days)
            
            disp('__________________________________________________________________')
            dn = simpleTokenize(days{iday}, '_');
            
            imagingFolder = dn{1};
            mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
            
            fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})
            
            
            %% Set vars
            
            signalCh = 2; % because you get A from channel 2, I think this should be always 2.
            pnev2load = [];
            [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
            % [pd, pnev_n] = fileparts(pnevFileName);
            [md,date_major] = fileparts(imfilename);
            cd(md) % r = repmat('%03d-', 1, length(mdfFileNumber)); r(end) = []; date_major = sprintf(['%s_', r], imagingFolder, mdfFileNumber);
            
            
            %% Load temp tuning vars
            
            clear fractNs_maxInEachFr_exc', 'fractNs_maxInEachFr_inh
            
            [pd, pnev_n] = fileparts(pnevFileName);
            postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
            
            [~,b] = fileparts(postName);
            fractName = ['tempTuning_', b(6:end)];
            
            load(fractName, 'fractNs_maxInEachFr_exc', 'fractNs_maxInEachFr_inh',...
                'eventI_initTone', 'eventI_stimOn', 'eventI_1stSideTry', 'eventI_reward',...
                'lenTrace_initTone', 'lenTrace_stimOn', 'lenTrace_1stSideTry', 'lenTrace_reward')
            
            
            %% Plot to compare temporal epoch tuning of exc and inh during the trial
            
            figure(f);
            subplot(ha(iday)); hold on
            plot(fractNs_maxInEachFr_exc, 'b')
            plot(fractNs_maxInEachFr_inh, 'r') % plot(smooth(fractNs_maxInEachFr_inh, 3), 'g')
            title(imagingFolder)
            
        end
        
        subplot(ha(1))
        xlabel('Frame')
        ylabel('Fract Ns with Max in each frame')
        legend('exc','inh')
        
        if savefigs
            savefig(f.Number, fullfile(dirn0, fign))
            print(gcf, '-dpdf', fullfile(dirn0, fign))
        end
    end
    
end


