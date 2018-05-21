%% Set number of neurons (exc, inh, unsure) and trials (all, corr, incorr) per session for each mouse

mice = {'fni16','fni17','fni18','fni19'};
signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
dirAllMice = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/fni_allMice/';


%%
Num_exc_inh_unsure_allMice = cell(1, length(mice));
Num_all_corr_incorr_allMice = cell(1, length(mice));

RT_goTochoiceTime_medAveTrs_allMice = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_easy_allMice = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_med_allMice = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_hard_allMice = cell(1, length(mice));

RT_goTochoiceTime_medAveTrs_allMice_easy = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_allMice_med = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_allMice_hard = cell(1, length(mice));

RT_goTochoiceTime_medAveTrs_allMice_easy_corr = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_allMice_med_corr = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_allMice_hard_corr = cell(1, length(mice));

RT_goTochoiceTime_medAveTrs_allMice_easy_incorr = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_allMice_med_incorr = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_allMice_hard_incorr = cell(1, length(mice));
    
RT_goTochoiceTime_medAveTrs_allMice_corr = cell(1, length(mice));
RT_goTochoiceTime_medAveTrs_allMice_incorr = cell(1, length(mice));

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
   
    
    %% Loop over days for each mouse

    Num_exc_inh_unsure = nan(length(days),3);
    Num_all_corr_incorr = nan(length(days),3);
    RT_goTochoiceTime_medAveTrs = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_easy = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_med = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_hard = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_easy_corr = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_med_corr = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_hard_corr = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_easy_incorr = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_med_incorr = nan(length(days),2);
    RT_goTochoiceTime_medAveTrs_hard_incorr = nan(length(days),2);    
    RT_goTochoiceTime_medAveTrs_incorr = nan(length(days),2);    
    RT_goTochoiceTime_medAveTrs_corr = nan(length(days),2);    
    
    % iday = 1;
    for iday = 1:length(days)

        dn = simpleTokenize(days{iday}, '_');

        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));

        fprintf('\n_________________  %s, day %d/%d (%s, sessions %s)  _________________\n', mouse, iday, length(days), imagingFolder, dn{2})

        [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
        [pd, pnev_n] = fileparts(pnevFileName);
        moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
        postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));

        %{
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
        %}

        
        %% Number of neurons per day
        
        load(moreName, 'inhibitRois_pix')
        Num_exc_inh_unsure(iday,:) = [sum(inhibitRois_pix==0), sum(inhibitRois_pix==1), sum(isnan(inhibitRois_pix))];
        
        
        %% Number of trials per day    
        
        load(postName, 'outcomes')
        Num_all_corr_incorr(iday,:) = [length(outcomes), sum(outcomes==1), sum(outcomes==0)];
        
        %% Reaction time: go tone to choice; median and mean across trials 
        
        load(postName, 'time1stSideTry', 'timeCommitCL_CR_Gotone', 'stimrate')
%         load(postName, 'timeStimOffset', 'timeSingleStimOffset', 'timeStimOnset')
%         figure; subplot(221),hist(timeStimOffset - timeStimOnset); hold on
%         subplot(222),hist(timeSingleStimOffset - timeStimOnset)
%         subplot(223),hist(time1stSideTry - timeStimOnset)

        rt = time1stSideTry - timeCommitCL_CR_Gotone;
        RT_goTochoiceTime_medAveTrs(iday,:) = [nanmedian(rt), nanmean(rt)];
        
        easyStim = abs(stimrate-16) >= 8;
        medStim = (abs(stimrate-16) > 3) & (abs(stimrate-16) < 8);
        hardStim = abs(stimrate-16) <= 3;
        
        rt_e = rt(easyStim);
        rt_m = rt(medStim);
        rt_h = rt(hardStim);        
        RT_goTochoiceTime_medAveTrs_easy(iday,:) = [nanmedian(rt_e), nanmean(rt_e)];
        RT_goTochoiceTime_medAveTrs_med(iday,:) = [nanmedian(rt_m), nanmean(rt_m)];
        RT_goTochoiceTime_medAveTrs_hard(iday,:) = [nanmedian(rt_h), nanmean(rt_h)];
        
        rt_e = rt(easyStim' & outcomes==1);
        rt_m = rt(medStim' & outcomes==1);
        rt_h = rt(hardStim' & outcomes==1);                
        RT_goTochoiceTime_medAveTrs_easy_corr(iday,:) = [nanmedian(rt_e), nanmean(rt_e)];
        RT_goTochoiceTime_medAveTrs_med_corr(iday,:) = [nanmedian(rt_m), nanmean(rt_m)];
        RT_goTochoiceTime_medAveTrs_hard_corr(iday,:) = [nanmedian(rt_h), nanmean(rt_h)];
        
        rt_e = rt(easyStim' & outcomes==0);
        rt_m = rt(medStim' & outcomes==0);
        rt_h = rt(hardStim' & outcomes==0);                
        RT_goTochoiceTime_medAveTrs_easy_incorr(iday,:) = [nanmedian(rt_e), nanmean(rt_e)];
        RT_goTochoiceTime_medAveTrs_med_incorr(iday,:) = [nanmedian(rt_m), nanmean(rt_m)];
        RT_goTochoiceTime_medAveTrs_hard_incorr(iday,:) = [nanmedian(rt_h), nanmean(rt_h)];
        
        rt_c = rt(outcomes==1);
        rt_i = rt(outcomes==0);
        RT_goTochoiceTime_medAveTrs_corr(iday,:) = [nanmedian(rt_c), nanmean(rt_c)];
        RT_goTochoiceTime_medAveTrs_incorr(iday,:) = [nanmedian(rt_i), nanmean(rt_i)];
        
    end
    
    Num_exc_inh_unsure_allMice{im} = Num_exc_inh_unsure;
    Num_all_corr_incorr_allMice{im} = Num_all_corr_incorr;
    
    RT_goTochoiceTime_medAveTrs_allMice{im} = RT_goTochoiceTime_medAveTrs;
    
    RT_goTochoiceTime_medAveTrs_allMice_easy{im} = RT_goTochoiceTime_medAveTrs_easy;
    RT_goTochoiceTime_medAveTrs_allMice_med{im} = RT_goTochoiceTime_medAveTrs_med;
    RT_goTochoiceTime_medAveTrs_allMice_hard{im} = RT_goTochoiceTime_medAveTrs_hard;

    RT_goTochoiceTime_medAveTrs_allMice_easy_corr{im} = RT_goTochoiceTime_medAveTrs_easy_corr;
    RT_goTochoiceTime_medAveTrs_allMice_med_corr{im} = RT_goTochoiceTime_medAveTrs_med_corr;
    RT_goTochoiceTime_medAveTrs_allMice_hard_corr{im} = RT_goTochoiceTime_medAveTrs_hard_corr;
    
    RT_goTochoiceTime_medAveTrs_allMice_easy_incorr{im} = RT_goTochoiceTime_medAveTrs_easy_incorr;
    RT_goTochoiceTime_medAveTrs_allMice_med_incorr{im} = RT_goTochoiceTime_medAveTrs_med_incorr;
    RT_goTochoiceTime_medAveTrs_allMice_hard_incorr{im} = RT_goTochoiceTime_medAveTrs_hard_incorr;
    
    RT_goTochoiceTime_medAveTrs_allMice_corr{im} = RT_goTochoiceTime_medAveTrs_corr;
    RT_goTochoiceTime_medAveTrs_allMice_incorr{im} = RT_goTochoiceTime_medAveTrs_incorr;
    
    
end

%%% Save a mat file for all mice
% save(fullfile(dirAllMice, 'numNeuronsTrials_eachSess_allMice'), 'Num_exc_inh_unsure_allMice', 'Num_all_corr_incorr_allMice')


%% Set mean and median numbers across days

med_exc_inh_unsure_all = nan(length(mice), 4);
med_ratio_inh_exc = nan(length(mice), 1);
ave_exc_inh_unsure_all = nan(length(mice), 4);
ave_ratio_inh_exc = nan(length(mice), 1);

med_all_corr_incorr = nan(length(mice), 3);
ave_all_corr_incorr = nan(length(mice), 3);

med_ratio_unsure_all = nan(length(mice), 1);
ave_ratio_unsure_all = nan(length(mice), 1);

for im = 1:length(mice)
    
    % Neurons
    
    % number of neurons: exc, inh, unsure, all 
    med_exc_inh_unsure_all(im,:) = [median(Num_exc_inh_unsure_allMice{im},1), median(sum(Num_exc_inh_unsure_allMice{im},2))];
    ave_exc_inh_unsure_all(im,:) = [mean(Num_exc_inh_unsure_allMice{im},1), mean(sum(Num_exc_inh_unsure_allMice{im},2))];

    % num inh / num exc
    med_ratio_inh_exc(im,:) = median(Num_exc_inh_unsure_allMice{im}(:,2) ./ Num_exc_inh_unsure_allMice{im}(:,1));
    ave_ratio_inh_exc(im,:) = mean(Num_exc_inh_unsure_allMice{im}(:,2) ./ Num_exc_inh_unsure_allMice{im}(:,1));
    
    
    % num unsure / num all
    med_ratio_unsure_all(im,:) = median(Num_exc_inh_unsure_allMice{im}(:,3) ./ sum(Num_exc_inh_unsure_allMice{im},2));
    ave_ratio_unsure_all(im,:) = mean(Num_exc_inh_unsure_allMice{im}(:,3) ./ sum(Num_exc_inh_unsure_allMice{im},2));
    
    
    %% Trials
    
    med_all_corr_incorr(im,:) = median(Num_all_corr_incorr_allMice{im},1);
    ave_all_corr_incorr(im,:) = mean(Num_all_corr_incorr_allMice{im},1);
    
    
end


%% Plot

savefigs = 1;
dirn0 = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/NumNeuronTrial_summary';

for im = 1:length(mice)
    
    fh = figure('position', [7   460   254   492], 'name', mice{im});
    
    subplot(211)
    boxplot([sum(Num_exc_inh_unsure_allMice{im},2), Num_exc_inh_unsure_allMice{im}], 'labels', {'all', 'exc', 'inh', 'unsure'})
    title('Neurons')
    set(gca, 'box', 'off', 'tickdir', 'out')

    subplot(212)
    boxplot(Num_all_corr_incorr_allMice{im}, 'labels', {'all', 'corr', 'incorr'})
    title('Trials')
    set(gca, 'box', 'off', 'tickdir', 'out')

    
    %%%%%%%%%%%%%%%%%% save figure %%%%%%%%%%%%%%%%%% 
    if savefigs        
        mouse = mice{im};
        dirfinal = fullfile(dirn0, [mice{im},'_numNsTrs']);
        
        savefig(fh, [dirfinal,'.fig'])
        print(fh, '-dpdf', dirfinal)
    end  
    
end


%% Learning figure: fraction correct and incorrect for each day

figure; 
for im = 1:length(mice)
    subplot(2,2,im); hold on; title(mice{im})
    plot(Num_all_corr_incorr_allMice{im}(:,2) ./ Num_all_corr_incorr_allMice{im}(:,1), 'k')
    plot(Num_all_corr_incorr_allMice{im}(:,3) ./ Num_all_corr_incorr_allMice{im}(:,1), 'color', rgb('orange'))    
    plot(Num_all_corr_incorr_allMice{im}(:,2) ./ Num_all_corr_incorr_allMice{im}(:,3), 'color', rgb('pink'))    
    
    if im==1
        legend('fractCorr','fractIncorr','corr/incorr')
    end
       
end


%% Learning figure: RT for each day, compare RT for easy, medium, hard trials.

%{
% below is not correct... you cant compare RT of easy with hard ACROSS
% days. you have to do it within a day
av_e_mh = nan(length(mice),2);
for im = 1:length(mice)
    av_e_mh(im,:) = [nanmean(RT_goTochoiceTime_medAveTrs_allMice_easy{im}(:,1)), ...
        nanmean([RT_goTochoiceTime_medAveTrs_allMice_med{im}(:,1); ...
        RT_goTochoiceTime_medAveTrs_allMice_hard{im}(:,1)])];
end
%}

figure; 
for im = 1:length(mice)
    subplot(2,2,im); hold on; title(mice{im})
    plot(RT_goTochoiceTime_medAveTrs_allMice_easy{im}(:,1), 'k')
    plot(RT_goTochoiceTime_medAveTrs_allMice_med{im}(:,1), 'g')
    plot(RT_goTochoiceTime_medAveTrs_allMice_hard{im}(:,1), 'r')
    xlabel('Session')
    ylabel('RT (ms)')
    set(gca, 'tickdir', 'out')
    if im==1
        legend('easy','medium','hard')
    end
    
    % Save figure:
%     '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/Behavior/reactionTime'
    
end


%% Learning figure: RT for each day, compare RT for corr, incorr trials

figure; 
for im = 1:length(mice)
    subplot(2,2,im); hold on; title(mice{im})

    plot(RT_goTochoiceTime_medAveTrs_allMice_corr{im}(:,1), 'k-')% 'r')    
    plot(RT_goTochoiceTime_medAveTrs_allMice_incorr{im}(:,1), 'k:')% 'color', rgb('pink'))
    
%     plot(RT_goTochoiceTime_medAveTrs_allMice_easy_corr{im}(:,1), 'k-')% 'k')
%     plot(RT_goTochoiceTime_medAveTrs_allMice_easy_incorr{im}(:,1), 'k:')% 'color', rgb('gray'))

%     plot(RT_goTochoiceTime_medAveTrs_allMice_med_corr{im}(:,1), 'k-')% 'g')
%     plot(RT_goTochoiceTime_medAveTrs_allMice_med_incorr{im}(:,1), 'k:')% 'color', rgb('lightgreen'))

%     plot(RT_goTochoiceTime_medAveTrs_allMice_hard_corr{im}(:,1), 'k-')% 'r')    
%     plot(RT_goTochoiceTime_medAveTrs_allMice_hard_incorr{im}(:,1), 'k:')% 'color', rgb('pink'))
    
    
    xlabel('Session')
    ylabel('RT (ms)')
    set(gca, 'tickdir', 'out')
    if im==1
        legend('corr', 'incorr')
    end       
end


