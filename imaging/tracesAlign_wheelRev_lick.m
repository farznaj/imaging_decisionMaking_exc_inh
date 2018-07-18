%{

% clear; close all
mouse = 'fni16'; %'fni17';
imagingFolder = '151028'; %'151015';
mdfFileNumber = [1,2];

%%
% close all
% best is to set the 2 vars below to 0 so u get times of events for all trials; later decide which ones to set to nan.
rmvTrsStimRateChanged = 0; % if 1, early-go-tone trials w stimRate categ different before and after go tone, will be excluded.
do = 0; % set to 1 when first evaluating a session, to get plots and save figs and vars.

normalizeSpikes = 1; % if 1, spikes trace of each neuron will be normalized by its max.
warning('Note you have set normalizeSpikes to 1!!!')

thbeg = 5; % n initial trials to exclude.
if strcmp(mouse, 'fni19') && strcmp(imagingFolder,'150918')
    thbeg = 7;
end
% normally rmvTrsStimRateChanged is 1, except for early sessions of
% training we have to set it to 0, otherwise lots of trials will be
% excluded bc go tone has happpened very early.

rmv_timeGoTone_if_stimOffset_aft_goTone = 0; % if 1, trials with stimOffset after goTone will be removed from timeGoTone (ie any analyses that aligns trials on the go tone)
rmv_time1stSide_if_stimOffset_aft_1stSide = 0; % if 1, trials with stimOffset after 1stSideTry will be removed from time1stSideTry (ie any analyses that aligns trials on the 1stSideTry)

evaluateEftyOuts = do; 
compareManual = do; % compare results with manual ROI extraction
plot_ave_noTrGroup = do; % Set to 1 when analyzing a session for the 1st time. Plots average imaging traces across all neurons and all trials aligned on particular trial events. Also plots average lick traces aligned on trial events.
save_aligned_traces = do; % save aligned_traces to postName
savefigs = do;

setInhibitExcit = 0; % if 1, inhibitory and excitatory neurons will be set unless inhibitRois is already saved in imfilename (in which case it will be loaded).
plotEftyAC1by1 = 0; % A and C for each component will be plotted 1 by 1 for evaluation of of Efty's results. 
frameLength = 1000/30.9; % sec.

[alldata, alldataSpikesGood, alldataDfofGood, goodinds, good_excit, good_inhibit, outcomes, allResp, allResp_HR_LR, ...
        trs2rmv, stimdur, stimrate, stimtype, cb, timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, ...
        timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, imfilename, pnevFileName] ....
   = wheelRevAlign(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength, save_aligned_traces, savefigs, rmvTrsStimRateChanged, thbeg);


%}

%%
function tracesAlign_wheelRev_lick(mouse, imagingFolder, mdfFileNumber, setInhibitExcit, rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, plot_ave_noTrGroup, evaluateEftyOuts, normalizeSpikes, compareManual, plotEftyAC1by1, frameLength, save_aligned_traces, savefigs, rmvTrsStimRateChanged, thbeg)


lickInds = [1, 2,3]; % Licks to analyze % 1: center lick, 2: left lick; 3: right lick
% lickInds = [1];


%% Set some initial variables.
    
excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;

if ~exist('thbeg', 'var')
    thbeg = 5; % n initial trials to exclude.
end
signalCh = 2;
pnev2load = []; %7 %4 % what pnev file to load (index based on sort from the latest pnev vile). Set [] to load the latest one.


%%
% set the names of imaging-related .mat file names.
% remember the last saved pnev mat file will be the pnevFileName
%{
signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
%}
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
% [~,b] = fileparts(imfilename);
% diary(['diary_',b])
disp(pnev_n)
% cd(fileparts(imfilename))

postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
% moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
aimf = matfile(imfilename);

% load alldata

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

begTrs = 1; % 1st trial of each session

load(imfilename, 'outputsDFT', 'badFrames', 'pmtOffFrames')



%%
%%%%%%%%%%%%%%%%%%%%%%%%% Load neural data, merge them into all_data, and assess trace quality %%%%%%%%%%%%%%%%%%%%%%%

spikes = nan(length(badFrames{1}),2); % we just need it so merging (below) doesn't give error! % assume we have 2 neruons
activity = spikes;
dFOF = spikes;
goodinds = ones(1,2);


%%

[nFrsSess, nFrsMov] = set_nFrsSess(mouse, imagingFolder, mdfFileNumber); % nFrsMov: for each session shows the number of frames in each tif file.
% cs_frmovs = [0, cumsum(cell2mat(nFrsMov))]; % cumsum of nFrsMov: shows number of frames per tif movie (includes all tif movies of mdfFileNumber). 


%%
%%%%%%%%%%%%%%%%%%%%%%%%% Merging imaging data to behavioral data %%%%%%%%%%%%%%%%%%%%%%%

if length(mdfFileNumber)==1
    
    load(imfilename, 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'trialCodeMissing') %, 'cs_frtrs')
   
%     figure('position', [68   331   254   139]); hold on; plot(framesPerTrial - diff(cs_frtrs))
%     xlabel('trial'), ylabel('framesPerTrial - diff(cs_frtrs)'), title('shows frame-dopped trials')

    
    if ~exist('trialCodeMissing', 'var'), trialCodeMissing = []; end  % for those days that you didn't save this var.
    
    if length(trialNumbers) ~= length(framesPerTrial)
        error('Investigate this. Make sure merging with imaging is done correctly.')
    end
    
    trialNumbers(isnan(framesPerTrial)) = [];
    frame1RelToStartOff(isnan(framesPerTrial)) = [];
    framesPerTrial(isnan(framesPerTrial)) = [];    
    
    
    cprintf('blue', 'Total number of imaged trials: %d\n', length(trialNumbers))
    if ~any(trialCodeMissing)
        cprintf('blue', 'All trials are triggered in MScan :)\n')
    else
        cprintf('blue', ['There are non-triggered trials in MScan! Trial(s):', repmat('%d ', 1, sum(trialCodeMissing)), '\n'], find(trialCodeMissing))
    end
    
    
    %% Remove trials at the end of alldata that are in behavior but not imaging.
    
    max_num_imaged_trs = max(length(framesPerTrial), max(trialNumbers)); % I think you should just do : max(trialNumbers)
    
    a = length(all_data) - max_num_imaged_trs;
    if a~=0
        cprintf('blue', 'Removing %i trials at the end of alldata bc imaging was aborted.\n', a)
    else
        fprintf('Removing no trials at the end of alldata bc imaging was not aborted earlier.\n')
    end
    
    all_data(max_num_imaged_trs+1: end) = []; % remove behavioral trials at the end that with no recording of imaging data.
    
    
    %% Merge imaging variables into all_data (before removing any trials): activity, dFOF, spikes

    minPts = 7000; %800;
    set2nan = 1; % if 1, in trials that were not imaged, set frameTimes, dFOF, spikes and activity traces to all nans (size: min(framesPerTrial) x #neurons).
    
    [all_data, mscanLag] = mergeActivityIntoAlldata_fn(all_data, activity, framesPerTrial, ...
        trialNumbers, frame1RelToStartOff, badFrames{signalCh}, pmtOffFrames{signalCh}, minPts, dFOF, spikes, set2nan);
    
    alldata = all_data;
    
    
    %% Set trs2rmv, stimrate, outcome and response side. You will set certain variables to NaN for trs2rmv (but you will never remove them from any arrays).
    
    load(imfilename, 'badAlignTrStartCode', 'trialStartMissing'); %, 'trialCodeMissing') % they get set in framesPerTrialStopStart3An_fn
    load(imfilename, 'trEndMissing', 'trEndMissingUnknown', 'trStartMissingUnknown') % Remember their indeces are on the imaged trials (not alldata). % do trialNumbers(trEndMissing) to find the corresponding indeces on alldata. 
    
    if ~exist('trEndMissing', 'var'), trEndMissing = []; end
    if ~exist('trEndMissingUnknown', 'var'), trEndMissingUnknown = []; end
    if ~exist('trStartMissingUnknown', 'var'), trStartMissingUnknown = []; end
    
    imagingFlg = 1;
    [trs2rmv, stimdur, stimrate, stimtype, cb] = setTrs2rmv_final(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, imagingFlg, badAlignTrStartCode, trialStartMissing, trialCodeMissing, trStartMissingUnknown, trEndMissing, trEndMissingUnknown, trialNumbers, rmvTrsStimRateChanged);
    trs2rmv(trs2rmv>length(alldata)) = [];
    
else
    
    % assing pmtOff and badFrames to each session
    % merge alldata with imaging for each session
    % set trs2rmv for each session
    multi_sess_set_vars
    
    if ~isprop(aimf, 'len_alldata_eachsess')
        save(imfilename, '-append', 'len_alldata_eachsess', 'framesPerTrial_alldata') % elements of these 2 arrays correspond to each other.
    end
    % it makes a lot of sense to save the above 2 vars for days with a
    % single session too. It will make things much easier! you need to have
    % a framesPerTrial array that corresponds to alldata, and has nan for
    % non-imaged trials. I think you just need to set a nan array same
    % length as the merged alldata (which I guess lacks the last trial),
    % and then fill it with framesPerTrial using trialNumbers as indeces.
end



%% In alldata, set good quality traces.

% alldataSpikesGood = cellfun(@(x)x(:, goodinds), {alldata.spikes}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
% alldataActivityGood = cellfun(@(x)x(:, goodinds), {alldata.activity}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.
alldataDfofGood = cellfun(@(x)x(:, goodinds), {alldata.dFOF}, 'uniformoutput', 0); % cell array, 1 x number of trials. Each cell is frames x units.







%%%%%% Prep for setting wheelRev and lick traces %%%%%%

%%

load(imfilename, 'trs2rmv')
load(postName, 'allResp_HR_LR', 'outcomes', 'stimrate', 'cb')
%{
load(postName, 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
        'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
        'timeReward', 'timeCommitIncorrResp', 'timeStop', ...
        'allResp_HR_LR', 'outcomes', 'stimrate', 'cb')
%}
   

%%
cprintf('blue', 'Plot average imaging traces and wheel revolution aligned on trial events\n')
evT = {'time1stSideTry'}; %{'1', 'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
%     'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
%     'timeReward', 'timeCommitIncorrResp', 'timeStop'};

unitFrame = 0; %1; % if 1, x axis will be in units of frames. If 0, x axis will be in units of time.


%%
% Set event times (ms) relative to when bcontrol starts sending the scope TTL. event times will be set to NaN for trs2rmv.
%{
[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv);
%}
close all

alldatanow = alldata; %alldata(trs2ana);
trs2rmvnow = trs2rmv; %find(ismember(find(trs2ana), trs2rmv));
doplots = 0;


%% Align traces, average them, and plot them.

%{
fn = sprintf('%s - Shades: stand error across trials (neuron-averaged traces of trials)', top);
% fn = sprintf('%s - Shades for rows 1 & 2: standard errors across neural traces (trial-averaged traces of neurons).  Shades for row 3: stand error across trials', top);
f = figure('name', fn, 'position', [680         587        1375         389]);
cnt = 0;
%}
for i = 1 %6 %ievents
    
%     cnt = cnt+1;

    % for licks we get the event times with scopeTTLOrigTime = 0; so they
    % change ... so we need to reload them here!
    load(postName, 'time1stSideTry')%'timeInitTone', 'timeStimOnset', 'timeStimOffset', 'timeCommitCL_CR_Gotone',...
%             'time1stSideTry', 'time1stCorrectTry', 'time1stIncorrectTry',...
%             'timeReward', 'timeCommitIncorrResp', 'timeStop')

    % set DF/F traces (C: temporal component) and wheel traces
    
    disp(['------- ', evT{i}, ' -------'])
    
    eventTime = eval(evT{i});      
    % Take only trials in trs2ana for analysis
%     if length(eventTime)>1
%         eventTime = eventTime(trs2ana);
%     end
    
    if ~iscell(eventTime) && isempty(eventTime), error('No trials!'), end    
    if (iscell(eventTime) && all(cellfun(@(x)all(isnan(x)), eventTime))) || (~iscell(eventTime) && all(isnan(eventTime)))
        warning('Choose outcome2ana and evT wisely! eg. you cannot choose outcome2ana=1 and evT="timeCommitIncorrResp"')
        error('Something wrong; eventTime is all NaNs. Could be due to improper choice of outcome2ana and evT{i}!')
    end
    
    traces = alldataDfofGood; % alldataSpikesGood; %  traces to be aligned.
%     traces = traces(trs2ana);
    
    
    %% Align wheel traces
    
    alignWheel = 1; %nan; % only align the wheel traces
    printsize = 1;
%     traces = [];
    
    [traceEventAlign, timeEventAlign, nvalidtrs, ...
        traceEventAlign_wheelRev, timeEventAlign_wheelRev, nvalidtrs_wheel] ...
        = avetrialAlign_noTrGroup(eventTime, traces, alldatanow, frameLength, trs2rmvnow, alignWheel, printsize, doplots, 1);
    
    
    %% Align lick traces
    
    % Get the time of events in ms relative to the start of each trial in bcontrol
    scopeTTLOrigTime = 0; % above we need event times with scopeTTLOrigTime = 1;
    
    [timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
        time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
        setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime, [], outcomes);
    
    eventTime = eval(evT{i});      
    
    
    %%%%%%%%%%%%%%%%%%    
    %     cprintf('blue', 'Plot average lick traces aligned on trial events.\n')    
    
    %%% Set the traces for licks (in time and frame resolution)
    [traces_lick_time, traces_lick_frame] = setLickTraces(alldata, outcomes);

    
    %%% Align    
    traces = traces_lick_time;
    traces = cellfun(@(x)ismember(x, lickInds), traces, 'uniformoutput', 0); % Only extract the licks that you want to analyze (center, left or right).
%     traces = traces(trs2ana);
    
    % it has lots of nans... nPrenPost is better... below
%     [traceEventAlign, timeEventAlign, nvalidtrs] = triggerAlignTraces(traces, eventTime);

    nPreFrames =[]; nPostFrames = [];
    [traceEventAlign_lick, timeEventAlign_lick, eventI_lick, nPreFrames, nPostFrames] ...
    = triggerAlignTraces_prepost(traces, round(eventTime), nPreFrames, nPostFrames); %, shiftTime, scaleTime, 1); % frames x units x trials        

    cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', 'firstSideTry', nPreFrames, nPostFrames)
    
    
    %% Save wheelRev vars
    
%     check npre and npost for wheelRev... how does it compare to how you get Al traces like firstSideTryAl ... it is like nPre , nPost = []        
    if i==1 %6
        firstSideTryAl_wheelRev.traces = traceEventAlign_wheelRev;
        firstSideTryAl_wheelRev.time = timeEventAlign_wheelRev;
        firstSideTryAl_wheelRev.eventI = nvalidtrs_wheel; 
        firstSideTryAl_wheelRev

        firstSideTryAl_lick.traces = traceEventAlign_lick;
        firstSideTryAl_lick.time = timeEventAlign_lick;
        firstSideTryAl_lick.eventI = eventI_lick; 
        firstSideTryAl_lick
        
        cprintf('magenta', 'Appending to postName wheelRev and lick traces aligned on choice\n')
        save(postName, '-append', 'firstSideTryAl_wheelRev', 'firstSideTryAl_lick')
    end

    
    
    %% Plots    
    
    if doplots
        
        % Plot wheel revolution
        
        tr00 = traceEventAlign_wheelRev;
        trt00 = timeEventAlign_wheelRev;
        
        figure; hold on
    %     subplot(3,length(ievents),length(ievents)*2+cnt), hold on

        tw = tr00;
    %     tw = tr00(:,:,bb);
        top = nanmean(tw,3); % average across trials
        tosd = nanstd(tw,[],3);
        tosd = tosd / sqrt(size(tw,3)); % plot se

        if unitFrame
            e = find(trt00 >= 0, 1);
            boundedline((1:length(top))-e, top, tosd, 'alpha')
        else
            boundedline(trt00, top, tosd, 'alpha')
        end
        %     plot(top)
        %     plot(trt00, top)

%         xl1 = find(nvalidtrs_wheel >= round(max(nvalidtrs_wheel)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
%         xl2 = find(nvalidtrs_wheel >= round(max(nvalidtrs_wheel)*4/4), 1, 'last');
        xl1 = 1; 
        xl2 = length(trt00); 
        
        if unitFrame
            xlim([xl1 xl2]-e)
        else
            xlim([trt00(xl1)  trt00(xl2)])
        end

        if unitFrame
            plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
            xlabel('Frame')
        else
            plot([frameLength/2  frameLength/2], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r')
            xlabel('Time (ms)')
        end
        ylim([min(top(xl1:xl2)-tosd(xl1:xl2))  max(top(xl1:xl2)+tosd(xl1:xl2))])
        ylabel('Wheel revolution')       
    
        
        %% Plot lick traces

        figure; hold on; %(f)
    %     subplot(2,length(evT),length(evT)*0+i), hold on

        av = nanmean(traceEventAlign_lick,3); % frames x units. (average across trials).
        top = nanmean(av,2); % average across neurons.
    %     tosd = nanstd(av, [], 2);
    %     tosd = tosd / sqrt(size(av, 2)); % plot se

        e = find(timeEventAlign_lick >= 0, 1);
    %     boundedline((1:length(top))-e, top, tosd, 'alpha')
        plot((1:length(top))-e, top)
        %     plot(top)
        %     plot(timeEventAlign, top)

    %     xl1 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'first'); % at least 3/4th of trials should contribute
    %     xl2 = find(nvalidtrs >= round(max(nvalidtrs)*4/4), 1, 'last');
        xl1 = 1; 
        xl2 = length(timeEventAlign_lick); 

        xlim([xl1 xl2]-e)

        plot([0 0], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')

        %{
        top = nanmean(nanmean(traceEventAlign,3),2); % average across trials and neurons.
        plot(top)
        %     plot(timeEventAlign, top)

        xl1 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'first'); % at least 3/4th of trials should contribute
        xl2 = find(nvalidtrs >= round(max(nvalidtrs)*3/4), 1, 'last');

        %     plot([0 0],[min(top(xl1:xl2)) max(top(xl1:xl2))], 'r:')
        e = find(timeEventAlign >= 0, 1);
        plot([e e], [min(top(xl1:xl2)) max(top(xl1:xl2))], 'r-.')

        xlim([xl1-100 xl2+100])
        %     xlim([xl1 xl2])
        %     xlim([e-1000 e+1000])
        %     xlim([timeEventAlign(xl1)  timeEventAlign(xl2)])
        %}
        if i==1
            xlabel('Time (ms)')
            ylabel('Fraction trials with licks')
        end
        if i>1
            title(evT{i}(5:end))
        else
            title(evT{i})        
        end
        ylabel('Fraction trials with licks')        
    end
    
end
    