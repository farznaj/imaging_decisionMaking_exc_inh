% Load vars (alrady saved in tracesAlign_wheelRev_lick),
% Average them across trials for each day
% Align them across days. 
% Saves vars of all mice in /home/farznaj/Shares/Churchland_hpc_home/space_managed_data/fni_allMice.
%
% Also make plots of all mice and save them in '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/Behavior/wheelRev_Lick'


%%
outcome2ana = 'corr'; % 'all'; % 'corr'
mice = {'fni16','fni17','fni18','fni19'};
thMinTrs = 10; % days with fewer than this number of trials wont be analyzed.

dira = '/home/farznaj/Shares/Churchland_hpc_home/space_managed_data/fni_allMice';
if strcmp(outcome2ana, 'corr')
    ocn = 'corr_allMice_';
else
    ocn = 'allOutcomes_allMice_';
end
nowStr = datestr(now, 'yymmdd-HHMMSS');


%% Average wheelRev and lick traces across trials for each day

wheelRev_traces_aveTrs_all = cell(1, length(mice));
lick_traces_aveTrs_all = cell(1, length(mice));
wheelRev_traces_aveTrs_all_lrResp = cell(1, length(mice));
lick_traces_aveTrs_all_lrResp = cell(1, length(mice));
wheelRev_traces_aveTrs_all_hrResp = cell(1, length(mice));
lick_traces_aveTrs_all_hrResp = cell(1, length(mice));
wheelRev_time_all = cell(1, length(mice));
lick_time_all = cell(1, length(mice));
wheelRev_eventI_all = cell(1, length(mice));
lick_eventI_all = cell(1, length(mice));
numTrs_all = cell(1, length(mice));
mnTrNum_allMice = cell(1, length(mice));
p_wheelRev_hr_lr_all = cell(1, length(mice));
p_lick_hr_lr_all = cell(1, length(mice));

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
    
    
    %% Average wheel revolution and lick traces across trials for each day

    wheelRev_traces_aveTrs_all{im} = cell(1, length(days));
    lick_traces_aveTrs_all{im} = cell(1, length(days));
    wheelRev_traces_aveTrs_all_lrResp{im} = cell(1, length(days));
    lick_traces_aveTrs_all_lrResp{im} = cell(1, length(days));
    wheelRev_traces_aveTrs_all_hrResp{im} = cell(1, length(days));
    lick_traces_aveTrs_all_hrResp{im} = cell(1, length(days));    
    wheelRev_time_all{im} = cell(1, length(days));
    lick_time_all{im} = cell(1, length(days));        
    wheelRev_eventI_all{im} = nan(1, length(days));
    lick_eventI_all{im} = nan(1, length(days));        
    numTrs_all{im} = nan(1, length(days));        
    mnTrNum_allMice{im} = nan(1, length(days));
    p_wheelRev_hr_lr_all{im} = cell(1, length(days));
    p_lick_hr_lr_all{im} = cell(1, length(days));
    
    for iday = 1:length(days)
        
        dn = simpleTokenize(days{iday}, '_');
        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
        fprintf('\n_________________  %s, day %d/%d (%s, sessions %s)  _________________\n', mouse, iday, length(days), imagingFolder, dn{2})
        
        signalCh = 2; % because you get A from channel 2, I think this should be always 2.
        pnev2load = [];

        [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
        [pd, pnev_n] = fileparts(pnevFileName);
        disp(pnev_n)

        postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));        
            
        
        %% Load wheelRev and lick traces for each day
        % remember vars below include all trials, if you want only correct
        % trials, load outcomes and set them!
        
        load(postName, 'firstSideTryAl_wheelRev', 'firstSideTryAl_lick', 'outcomes', 'allResp_HR_LR')
        wheelTraces = firstSideTryAl_wheelRev.traces;
        lickTraces = firstSideTryAl_lick.traces;

        % if you don't want to include change-of-mind trials run below. This is done to use the same days as those on which class accur was
        % computed ... alternatively you can download 'corr_ipsi_contra' from roc vars saved in the analysis folder of each mouse.
        load(postName, 'time1stCorrectTry', 'time1stIncorrectTry', 'timeCommitIncorrResp');
        set_change_of_mind_trs
        outcomes(trs_com) = nan; 
        
        
        if strcmp(outcome2ana, 'corr')
            wheelTraces = wheelTraces(:,:,outcomes==1); 
            lickTraces = lickTraces(:,:,outcomes==1);
            allResp_HR_LR = allResp_HR_LR(outcomes==1);
        end
        
        numTrs_all{im}(iday) = sum(~isnan(wheelTraces(1,1,:)));                    
        mnTrNum_allMice{im}(iday) = min(sum(allResp_HR_LR==1), sum(allResp_HR_LR==0)); 
        disp([sum(allResp_HR_LR==1), sum(allResp_HR_LR==0)])
        
        
        %% Average across all trials        
        
        wheelAvtr = nanmean(wheelTraces,3);
        lickAvtr = nanmean(lickTraces,3);        
        
        % Average across only left or only right trials
        %%% Set licks and wheelRev for left vs right choice trials separately... we want to see if the pattern of left vs right choice licks (wheelRev) varies with trainin
        wheelAvtr_lrResp = nanmean(wheelTraces(:,:,allResp_HR_LR==0), 3); % 1 for HR choice, 0 for LR choice.
        wheelAvtr_hrResp = nanmean(wheelTraces(:,:,allResp_HR_LR==1), 3); % 1 for HR choice, 0 for LR choice.
        
        lickAvtr_lrResp = nanmean(lickTraces(:,:,allResp_HR_LR==0), 3);
        lickAvtr_hrResp = nanmean(lickTraces(:,:,allResp_HR_LR==1), 3);
        
        
        %%% ttest: hr vs lr
        a = squeeze(wheelTraces(:,:,allResp_HR_LR==0));
        b = squeeze(wheelTraces(:,:,allResp_HR_LR==1));        
        [~,p_wheelRev_hr_lr_all{im}{iday}] = ttest2(a',b');
        
        a = squeeze(lickTraces(:,:,allResp_HR_LR==0));
        b = squeeze(lickTraces(:,:,allResp_HR_LR==1));        
        [~,p_lick_hr_lr_all{im}{iday}] = ttest2(a',b');
        
        
        %%%% Keep vars for all days and all mice        
        % traces
        wheelRev_traces_aveTrs_all{im}{iday} = wheelAvtr;
        lick_traces_aveTrs_all{im}{iday} = lickAvtr;
        
        wheelRev_traces_aveTrs_all_lrResp{im}{iday} = wheelAvtr_lrResp;
        lick_traces_aveTrs_all_lrResp{im}{iday} = lickAvtr_lrResp;
        
        wheelRev_traces_aveTrs_all_hrResp{im}{iday} = wheelAvtr_hrResp;
        lick_traces_aveTrs_all_hrResp{im}{iday} = lickAvtr_hrResp;
        
        % time
        wheelRev_time_all{im}{iday} = firstSideTryAl_wheelRev.time;
        lick_time_all{im}{iday} = firstSideTryAl_lick.time;
        % eventI
        wheelRev_eventI_all{im}(iday) = firstSideTryAl_wheelRev.eventI;
        lick_eventI_all{im}(iday) = firstSideTryAl_lick.eventI;
        

    end
end


for im=1:4; sum(numTrs_all{im} < thMinTrs), end
for im=1:4; sum(mnTrNum_allMice{im} < thMinTrs), end

no


%% Align traces of all days for each mouse

time_aligned_wheelRev_allMice = cell(1, length(mice));
time_aligned_lick_allMice = cell(1, length(mice));

wheelRev_aligned_allMice = cell(1, length(mice));
wheelRev_aligned_allMice_lrResp = cell(1, length(mice));
wheelRev_aligned_allMice_hrResp = cell(1, length(mice));

lick_aligned_allMice = cell(1, length(mice));
lick_aligned_allMice_lrResp = cell(1, length(mice));
lick_aligned_allMice_hrResp = cell(1, length(mice));

nPreMin_wheelRev_allMice = nan(1, length(mice));
nPreMin_lick_allMice = nan(1, length(mice));

p_wheelRev_hr_lr_aligned_allMice = cell(1, length(mice));
p_lick_hr_lr_aligned_allMice = cell(1, length(mice));

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
    
        
    %% Set nPre and nPost for aligning traces 
    
    %%%% wheelRev
    nPost = nan(1,length(days));
    for iday = 1:length(days)        
        lentrace = length(wheelRev_traces_aveTrs_all{im}{iday});
        evi = wheelRev_eventI_all{im}(iday);        
        nPost(iday) = lentrace - evi;
    end    
    nPostMin_wheelRev = min(nPost);
    nPreMin_wheelRev = min(wheelRev_eventI_all{im})-1;
    disp([nPreMin_wheelRev , nPostMin_wheelRev])
    
    %%%% lick
    nPost = nan(1,length(days));
    for iday = 1:length(days)        
        lentrace = length(lick_traces_aveTrs_all{im}{iday});
        evi = lick_eventI_all{im}(iday);        
        nPost(iday) = lentrace - evi;
    end    
    nPostMin_lick = min(nPost);
    nPreMin_lick = min(lick_eventI_all{im})-1;
    disp([nPreMin_lick , nPostMin_lick])
    
    
    %% Set time_aligned for each mouse
    
    time_aligned_wheelRev = sort([(1:2:2*nPreMin_wheelRev)*-5 , (1:2:2*nPostMin_wheelRev)*5]); % every 10 ms
    time_aligned_lick = sort([(1:nPreMin_lick)*-1 , 0, (1:nPostMin_lick)]); % every 1 ms
    
    disp(size(time_aligned_wheelRev))
    disp(size(time_aligned_lick))
    
    
    %% Align traces of all days on the common eventI
    
    % work on this: exclude days with too few trials.
            
    wheelRev_aligned = nan(length(days), length(time_aligned_wheelRev)); 
    wheelRev_aligned_lrResp = nan(length(days), length(time_aligned_wheelRev)); 
    wheelRev_aligned_hrResp = nan(length(days), length(time_aligned_wheelRev)); 
    
    lick_aligned = nan(length(days), length(time_aligned_lick));
    lick_aligned_lrResp = nan(length(days), length(time_aligned_lick));
    lick_aligned_hrResp = nan(length(days), length(time_aligned_lick));
    
    p_wheelRev_hr_lr_aligned = nan(length(days), length(time_aligned_wheelRev));
    p_lick_hr_lr_aligned = nan(length(days), length(time_aligned_lick));
    
    for iday = 1:length(days)
%         if mnTrNum(iday,:) >= thMinTrs                        
        wheelRev_aligned(iday,:) = wheelRev_traces_aveTrs_all{im}{iday}(wheelRev_eventI_all{im}(iday)-nPreMin_wheelRev : wheelRev_eventI_all{im}(iday)+nPostMin_wheelRev-1);             
        wheelRev_aligned_lrResp(iday,:) = wheelRev_traces_aveTrs_all_lrResp{im}{iday}(wheelRev_eventI_all{im}(iday)-nPreMin_wheelRev : wheelRev_eventI_all{im}(iday)+nPostMin_wheelRev-1);             
        wheelRev_aligned_hrResp(iday,:) = wheelRev_traces_aveTrs_all_hrResp{im}{iday}(wheelRev_eventI_all{im}(iday)-nPreMin_wheelRev : wheelRev_eventI_all{im}(iday)+nPostMin_wheelRev-1);             
        
        lick_aligned(iday,:) = lick_traces_aveTrs_all{im}{iday}(lick_eventI_all{im}(iday)-nPreMin_lick : lick_eventI_all{im}(iday)+nPostMin_lick); 
        lick_aligned_lrResp(iday,:) = lick_traces_aveTrs_all_lrResp{im}{iday}(lick_eventI_all{im}(iday)-nPreMin_lick : lick_eventI_all{im}(iday)+nPostMin_lick); 
        lick_aligned_hrResp(iday,:) = lick_traces_aveTrs_all_hrResp{im}{iday}(lick_eventI_all{im}(iday)-nPreMin_lick : lick_eventI_all{im}(iday)+nPostMin_lick);         
        
        p_wheelRev_hr_lr_aligned(iday,:) = p_wheelRev_hr_lr_all{im}{iday}(wheelRev_eventI_all{im}(iday)-nPreMin_wheelRev : wheelRev_eventI_all{im}(iday)+nPostMin_wheelRev-1);
        p_lick_hr_lr_aligned(iday,:) = p_lick_hr_lr_all{im}{iday}(lick_eventI_all{im}(iday)-nPreMin_lick : lick_eventI_all{im}(iday)+nPostMin_lick);        
%         end
    end    
    
    
    %% Keep vars of all mice
    
    time_aligned_wheelRev_allMice{im} = time_aligned_wheelRev;    
    nPreMin_wheelRev_allMice(im) = nPreMin_wheelRev;
    wheelRev_aligned_allMice{im} = wheelRev_aligned;
    wheelRev_aligned_allMice_lrResp{im} = wheelRev_aligned_lrResp;
    wheelRev_aligned_allMice_hrResp{im} = wheelRev_aligned_hrResp;
    
    time_aligned_lick_allMice{im} = time_aligned_lick;
    nPreMin_lick_allMice(im) = nPreMin_lick;
    lick_aligned_allMice{im} = lick_aligned;
    lick_aligned_allMice_lrResp{im} = lick_aligned_lrResp;
    lick_aligned_allMice_hrResp{im} = lick_aligned_hrResp;
    
    p_wheelRev_hr_lr_aligned_allMice{im} = p_wheelRev_hr_lr_aligned;
    p_lick_hr_lr_aligned_allMice{im} = p_lick_hr_lr_aligned;
    
end



%% Make a copy of wheelRev and lick traces before subtracting the baseline

wheelRev_aligned_allMice0 = wheelRev_aligned_allMice;
wheelRev_aligned_allMice0_lrResp = wheelRev_aligned_allMice_lrResp;
wheelRev_aligned_allMice0_hrResp = wheelRev_aligned_allMice_hrResp;

lick_aligned_allMice0 = lick_aligned_allMice;
lick_aligned_allMice0_lrResp = lick_aligned_allMice_lrResp;
lick_aligned_allMice0_hrResp = lick_aligned_allMice_hrResp;


%% Postprocess wheelRev: so it starts at 0 (in the choice-aligned traces) and is in units of mm (ie distance travelled from the begining of choice-aligned trace)
% Make all day traces start from the same baseline; also flip the traces, so they go positive. (currently they go negative if the mouse runs).
% Also turn wheel revolution to travelled distance (mm).

periph = 455; % 455mm is the periphary of the wheel.

for im = 1:length(mice)
    bl = 0; % nanmin(wheelRev_aligned_allMice0{im}(:,1));
    bld = wheelRev_aligned_allMice0{im}(:,1) - bl;    
    wheelRev_aligned_allMice{im} = -(wheelRev_aligned_allMice0{im} - bld) * periph; 

    bl = 0; % nanmin(wheelRev_aligned_allMice0{im}(:,1));
    bld = wheelRev_aligned_allMice0_lrResp{im}(:,1) - bl;    
    wheelRev_aligned_allMice_lrResp{im} = -(wheelRev_aligned_allMice0_lrResp{im} - bld) * periph; 

    bl = 0; % nanmin(wheelRev_aligned_allMice0{im}(:,1));
    bld = wheelRev_aligned_allMice0_hrResp{im}(:,1) - bl;    
    wheelRev_aligned_allMice_hrResp{im} = -(wheelRev_aligned_allMice0_hrResp{im} - bld) * periph; 
    
end



%% Set speed (units: mm/sec) of mouse running on the wheel ... compute it in 100ms bins (ie every 10 bin)
% first downsample the wheelRev trace, then compute speed on it.

downSampBin = 10; % number of bins to downsample (ie average every 10 bins)
binLength = 10; % ms; each bin of wheelRev data is 10 ms.

[speed_d_allMice, wheelRev_d_allMice, td_allMice, nPreMin_wheelRev_allMice_ds] = downsampWheelRev(wheelRev_aligned_allMice, time_aligned_wheelRev_allMice, nPreMin_wheelRev_allMice, downSampBin, binLength, mice);
[speed_d_allMice_lrResp, wheelRev_d_allMice_lrResp] = downsampWheelRev(wheelRev_aligned_allMice_lrResp, time_aligned_wheelRev_allMice, nPreMin_wheelRev_allMice, downSampBin, binLength, mice);
[speed_d_allMice_hrResp, wheelRev_d_allMice_hrResp] = downsampWheelRev(wheelRev_aligned_allMice_hrResp, time_aligned_wheelRev_allMice, nPreMin_wheelRev_allMice, downSampBin, binLength, mice);



%% Save vars

fn = ['wheelRev_lick_', ocn, nowStr];
fnam = fullfile(dira, fn);

save(fnam, 'time_aligned_lick_allMice', 'nPreMin_lick_allMice', 'lick_aligned_allMice', 'lick_aligned_allMice_lrResp', 'lick_aligned_allMice_hrResp', ...
    'time_aligned_wheelRev_allMice', 'nPreMin_wheelRev_allMice', 'wheelRev_aligned_allMice', 'wheelRev_aligned_allMice_lrResp', 'wheelRev_aligned_allMice_hrResp', ...
    'td_allMice', 'nPreMin_wheelRev_allMice_ds', 'speed_d_allMice', 'speed_d_allMice_lrResp', 'speed_d_allMice_hrResp', ...
    'wheelRev_d_allMice', 'wheelRev_d_allMice_lrResp', 'wheelRev_d_allMice_hrResp', ...
    'p_wheelRev_hr_lr_aligned_allMice', 'p_lick_hr_lr_aligned_allMice', ...
    'downSampBin', 'numTrs_all', 'mnTrNum_allMice')



