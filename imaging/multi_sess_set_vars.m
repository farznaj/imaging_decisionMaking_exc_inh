%% Compute number of frames per tif file (nFrsMov) and also for each session (nFrsSess) using badFramesTif mat files.

% mouse = 'fni17';
% imagingFolder = '151022';
% mdfFileNumber = [1,2]; % or tif major
PP = struct; PP.signalCh = 2; % The only required field of P; % channel whose signal activity you want to analyze (normally 2 for gcamp channel). %
params = writeCaProcessParams('', mouse, imagingFolder, mdfFileNumber, PP);


warning('off', 'MATLAB:MatFile:OlderFormat')
tifNumsOrig = params.tifNums;

% compute number of frames per movie for each session (ie each mdf file aka tif major)
sess = unique(tifNumsOrig(:,2));
nFrsMov = cell(1, length(sess));
date_major_se = cell(1, length(sess));

for ise = 1:length(sess)
    s = tifNumsOrig(:,2)==sess(ise);
    tifMinor = unique(tifNumsOrig(s, 3))';
    date_major_se{ise} = sprintf('%06d_%03d', tifNumsOrig(find(s,1), 1:2)); % if there is only one mdfFile in tifNums, date_major_se will be same as date_major.
    for itm = 1:length(tifMinor)
        if ~any(params.oldTifName)
            a = dir(fullfile(params.tifFold, [date_major_se{ise}, '_00', num2str(tifMinor(itm)), '.mat']));
        elseif all(params.oldTifName)
            a = dir(fullfile(params.tifFold, [date_major_se{ise}, '_0', num2str(tifMinor(itm)), '.mat']));
        else
            disp('some tif minors are named 00 and some are named 0... work on your codes.')
        end
        
        a = matfile(a.name);
        nFrsMov{ise}(itm) = cellfun(@length, a.badFramesTif(1,1));
    end
end

nFrsSess = cellfun(@sum, nFrsMov);
fprintf(['# frames per session: ', repmat('%i ', 1, length(nFrsSess)), 'Total: ', '%i', '\n'], nFrsSess, sum(nFrsSess))


%% Set activity, dFOF and spikes for each session.

activity_sess = cell(1, length(mdfFileNumber));
dFOF_sess = cell(1, length(mdfFileNumber));
spikes_sess = cell(1, length(mdfFileNumber));

for ise = 1:length(mdfFileNumber)
    cs_nfrs_sess = cumsum([0 nFrsSess]);
    
    activity_sess{ise} = activity(cs_nfrs_sess(ise)+1: cs_nfrs_sess(ise+1), :);
    dFOF_sess{ise} = dFOF(cs_nfrs_sess(ise)+1: cs_nfrs_sess(ise+1), :);
    spikes_sess{ise} = spikes(cs_nfrs_sess(ise)+1: cs_nfrs_sess(ise+1), :);
end


%% Set imfilename for each session

imfilename_sess = cell(1, length(mdfFileNumber));

for ise = 1:length(mdfFileNumber)
    imfilename_sess{ise} = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber(ise), signalCh, pnev2load);
end
celldisp(imfilename_sess)


%%

pmtOffFrames_sess = cell(1, length(mdfFileNumber));
badFrames_sess = cell(1, length(mdfFileNumber));
all_data_sess = cell(1, length(mdfFileNumber));
framesPerTrial_sess = cell(1, length(mdfFileNumber));
trialNumbers_sess = cell(1, length(mdfFileNumber));
frame1RelToStartOff_sess = cell(1, length(mdfFileNumber));
trialCodeMissing_sess = cell(1, length(mdfFileNumber));

for ise = 1:length(mdfFileNumber)
    
    % Load alldata for each session
    [all_data_sess{ise}, ~] = loadBehavData(alldata_fileNames(mdfFileNumber(ise)));
    
    
    %%
    load(imfilename_sess{ise}, 'pmtOffFrames', 'badFrames')
    
    pmtOffFrames_sess{ise} = pmtOffFrames;
    badFrames_sess{ise} = badFrames;
    
    
    %%
    
    load(imfilename_sess{ise}, 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'trialCodeMissing')
    
    if ~exist('trialCodeMissing', 'var'), trialCodeMissing = []; end  % for those days that you didn't save this var.
    
    if length(trialNumbers) ~= length(framesPerTrial)
        error('Investigate this. Make sure merging with imaging is done correctly.')
    end
    
    trialNumbers(isnan(framesPerTrial)) = [];
    frame1RelToStartOff(isnan(framesPerTrial)) = [];
    framesPerTrial(isnan(framesPerTrial)) = [];
    
    
    framesPerTrial_sess{ise} = framesPerTrial;
    trialNumbers_sess{ise} = trialNumbers;
    frame1RelToStartOff_sess{ise} = frame1RelToStartOff;
    trialCodeMissing_sess{ise} = trialCodeMissing;
    
    fprintf('Total number of imaged trials: %d\n', length(trialNumbers))
    if ~any(trialCodeMissing)
        cprintf('blue', 'All trials are triggered in MScan :)\n')
    else
        cprintf('blue', ['There are non-triggered trials in MScan! Trial(s) ', repmat('%d ',1,sum(trialCodeMissing)), '\n'], find(trialCodeMissing))
    end
    
    
    %% Remove trials at the end of alldata that are in behavior but not imaging.
    
    max_num_imaged_trs = max(length(framesPerTrial_sess{ise}), max(trialNumbers_sess{ise})); % I think you should just do : max(trialNumbers_sess{ise})
    
    a = length(all_data_sess{ise}) - max_num_imaged_trs;
    if a~=0
        fprintf('Removing %i trials at the end of alldata bc imaging was aborted.\n', a)
    else
        fprintf('Removing no trials at the end of alldata bc imaging was not aborted earlier.\n')
    end
    
    all_data_sess{ise}(max_num_imaged_trs+1: end) = []; % remove behavioral trials at the end that with no recording of imaging data.
    
    
end

len_alldata_eachsess = cellfun(@length, all_data_sess)
cs_alldata = cumsum([0 len_alldata_eachsess]);
sum(len_alldata_eachsess)


%%% Set framesPerTrial for all sessions taking into account the missed trials.

framesPerTrial = NaN(1, sum(len_alldata_eachsess));
for ise = 1:length(mdfFileNumber)
    framesPerTrial(trialNumbers_sess{ise} + cs_alldata(ise)) = framesPerTrial_sess{ise};
end


%% Merge behavior and imaging

minPts = 7000; % 800;
set2nan = 1; % if 1, in trials that were not imaged, set frameTimes, dFOF, spikes and activity traces to all nans (size: min(framesPerTrial) x #neurons).

for ise = 1:length(mdfFileNumber)
    [all_data_sess{ise}, mscanLag] = mergeActivityIntoAlldata_fn(all_data_sess{ise}, activity_sess{ise}, framesPerTrial_sess{ise}, ...
        trialNumbers_sess{ise}, frame1RelToStartOff_sess{ise}, badFrames_sess{ise}{signalCh}, pmtOffFrames_sess{ise}{signalCh}, ...
        minPts, dFOF_sess{ise}, spikes_sess{ise}, set2nan);
end

alldata = cell2mat(all_data_sess);


%% Set trs2rmv, stimrate, outcome and response side. You will set certain variables to NaN for trs2rmv (but you will never remove them from any arrays).

imagingFlg = 1;
begTrs = 1; % 1st trial of each session
% begTrs = [0 cumsum(trials_per_session)]+1;
% begTrs = begTrs(1:end-1);

trs2rmv = []; stimdur = []; stimrate = []; stimtype = [];

for ise = 1:length(mdfFileNumber)
    load(imfilename_sess{ise}, 'badAlignTrStartCode', 'trialStartMissing'); %, 'trialCodeMissing') % they get set in framesPerTrialStopStart3An_fn
    
    [trs2rmv_sess, stimdur_sess, stimrate_sess, stimtype_sess, cb] = setTrs2rmv_final(all_data_sess{ise}, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, imagingFlg, badAlignTrStartCode, trialStartMissing, trialCodeMissing_sess{ise});
    
    trs2rmv = [trs2rmv; trs2rmv_sess + cs_alldata(ise)];
    stimdur = [stimdur; stimdur_sess];
    stimrate = [stimrate; stimrate_sess];
    stimtype = [stimtype; stimtype_sess];
    
end
