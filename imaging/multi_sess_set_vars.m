% set_nFrsSess % run this to get nFrsSess

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


%% Setting pmtOffFrames and badFrames for each session

% load(imfilename, 'pmtOffFrames', 'badFrames')
% pmtOffFrames0 = pmtOffFrames;
% badFrames0 = badFrames;
csnow = [0, cumsum(nFrsSess)];

pmtOffFrames_sess = cell(1, length(mdfFileNumber));
badFrames_sess = cell(1, length(mdfFileNumber));
all_data_sess = cell(1, length(mdfFileNumber));
framesPerTrial_sess = cell(1, length(mdfFileNumber));
trialNumbers_sess = cell(1, length(mdfFileNumber));
frame1RelToStartOff_sess = cell(1, length(mdfFileNumber));
trialCodeMissing_sess = cell(1, length(mdfFileNumber));

for ise = 1:length(mdfFileNumber)
    
    disp('_____________________')
    % Load alldata for each session
    [all_data_sess{ise}, ~] = loadBehavData(alldata_fileNames(mdfFileNumber(ise)));

    
    %%
    clear cs_frtrs pmtOffFrames badFrames trialCodeMissing
    load(imfilename_sess{ise}, 'pmtOffFrames', 'badFrames')
    
%     if exist('pmtOffFrames', 'var')
%         error('How come? shoulnt it be saved under the combined mat file and not in individual mat files?')
%     end
    if ~exist('pmtOffFrames', 'var') % Should be always the case if it is the 1st time that you run this code on a multi-session day. Because in multi-session case, badFrames and pmtOffFrames get saved in the combined (_001-002) mat file!
%         fprintf('pmtOffFrames does not exist. Setting it to all false and saving it!\n')
%         load(imfilename_sess{ise}, 'cs_frtrs')       
        
        pmtOffFrames = cell(1,2); 
        pmtOffFrames{1} = pmtOffFrames0{1}(csnow(ise)+1 : csnow(ise+1));
%         pmtOffFrames{1} = false(cs_frtrs(end),1); 
        pmtOffFrames{2} = pmtOffFrames{1};         
%         badFrames = pmtOffFrames;

        badFrames = cell(1,2); 
        badFrames{1} = badFrames0{1}(csnow(ise)+1 : csnow(ise+1));
%         pmtOffFrames{1} = false(cs_frtrs(end),1); 
        badFrames{2} = badFrames{1}; 
        
        
        save(imfilename_sess{ise}, 'pmtOffFrames', 'badFrames', '-append')
    end
    pmtOffFrames_sess{ise} = pmtOffFrames;
    badFrames_sess{ise} = badFrames;    
    
    
    %%
    
    load(imfilename_sess{ise}, 'framesPerTrial', 'trialNumbers', 'frame1RelToStartOff', 'trialCodeMissing')
    
    if ~exist('trialCodeMissing', 'var'), trialCodeMissing = []; end  % for those days that you didn't save this var.
    
    if length(trialNumbers) ~= length(framesPerTrial)
        error('Investigate this. Make sure merging with imaging is done correctly.')
    end
    
    trialNumbers(isnan(framesPerTrial)) = [];
    if length(frame1RelToStartOff) >= length(framesPerTrial)
        frame1RelToStartOff(isnan(framesPerTrial)) = [];
    end
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
        cprintf('blue', 'Removing %i trials at the end of alldata bc imaging was aborted.\n', a)
    else
        fprintf('Removing no trials at the end of alldata bc imaging was not aborted earlier.\n')
    end
    
    all_data_sess{ise}(max_num_imaged_trs+1: end) = []; % remove behavioral trials at the end that with no recording of imaging data.
    
    
end

len_alldata_eachsess = cellfun(@length, all_data_sess);
cprintf('blue', ['length of alldata of each session: ', repmat('%i  ', 1, length(len_alldata_eachsess)), 'Total: %i\n'], [len_alldata_eachsess, sum(len_alldata_eachsess)])
cs_alldata = cumsum([0 len_alldata_eachsess]);


%%% Set framesPerTrial for all sessions taking into account the missed trials.

% IMPORTANT: Remember framesPerTrial defined below is not a concatenation of
% framesPerTrial arrays of all sessions. Instead it corresponds to all_data
% of all sessions. So if a trial is not scanned, just how it appears in
% all_data, it will also appear in framesPerTrial below, but its value will
% be nan.

framesPerTrial = NaN(1, sum(len_alldata_eachsess));
for ise = 1:length(mdfFileNumber)
    ii = trialNumbers_sess{ise};
    if sum(isnan(ii)) > 0        
        if sum(isnan(ii))==1
            disp('the very unusual case that trialCode is missing but still some frames are recorded. trialNumbers was set to nan')
        elseif sum(isnan(ii)) > 1
            error('the very unusual case that trialCode is missing but still some frames are recorded. trialNumbers was set to nan')
        end
        ii(isnan(ii)) = ii(find(isnan(ii))+1) - 1;
    end
%     a{ise} = ii + cs_alldata(ise);
    framesPerTrial(ii + cs_alldata(ise)) = framesPerTrial_sess{ise};
end
framesPerTrial_alldata = framesPerTrial;


%% Merge behavior and imaging

minPts = 7000; % 800;
set2nan = 1; % if 1, in trials that were not imaged, set frameTimes, dFOF, spikes and activity traces to all nans (size: min(framesPerTrial) x #neurons).
mscanLag_sess = cell(1,length(mdfFileNumber));

for ise = 1:length(mdfFileNumber)
    disp('_____________________')
    fprintf('Merging session %i ... \n', ise)
    [all_data_sess{ise}, mscanLag_sess{ise}] = mergeActivityIntoAlldata_fn(all_data_sess{ise}, activity_sess{ise}, framesPerTrial_sess{ise}, ...
        trialNumbers_sess{ise}, frame1RelToStartOff_sess{ise}, badFrames_sess{ise}{signalCh}, pmtOffFrames_sess{ise}{signalCh}, ...
        minPts, dFOF_sess{ise}, spikes_sess{ise}, set2nan);
end

alldata = cell2mat(all_data_sess);
mscanLag = cell2mat(mscanLag_sess);


%% Set trs2rmv, stimrate, outcome and response side. You will set certain variables to NaN for trs2rmv (but you will never remove them from any arrays).

disp('=== Setting Trial Numbers ===')
imagingFlg = 1;
begTrs = 1; % 1st trial of each session
% begTrs = [0 cumsum(trials_per_session)]+1;
% begTrs = begTrs(1:end-1);

trs2rmv = []; stimdur = []; stimrate = []; stimtype = [];

for ise = 1:length(mdfFileNumber)
    disp('_____________________')
    fprintf('\tsession %i ... \n', ise)
    
    clear trEndMissing trEndMissingUnknown trStartMissingUnknown % you didnt have this until 8/16/17... so it means all days analyzed prior to this date may have excluded some trials for sessions 2-end if trialEndMissing etc was not saved for them but existed for the earlier session!
    load(imfilename_sess{ise}, 'trialNumbers')
    load(imfilename_sess{ise}, 'badAlignTrStartCode', 'trialStartMissing'); %, 'trialCodeMissing') % they get set in framesPerTrialStopStart3An_fn
    load(imfilename_sess{ise}, 'trEndMissing', 'trEndMissingUnknown', 'trStartMissingUnknown')   
    
    if ~exist('trEndMissing', 'var'), trEndMissing = []; end
    if ~exist('trEndMissingUnknown', 'var'), trEndMissingUnknown = []; end
    if ~exist('trStartMissingUnknown', 'var'), trStartMissingUnknown = []; end
    
    [trs2rmv_sess, stimdur_sess, stimrate_sess, stimtype_sess, cb] = setTrs2rmv_final(all_data_sess{ise}, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, imagingFlg, badAlignTrStartCode, trialStartMissing, trialCodeMissing_sess{ise}, trStartMissingUnknown, trEndMissing, trEndMissingUnknown, trialNumbers, rmvTrsStimRateChanged);
    trs2rmv_sess(trs2rmv_sess > length(all_data_sess{ise})) = [];
    
    trs2rmv = [trs2rmv; trs2rmv_sess + cs_alldata(ise)];
    stimdur = [stimdur; stimdur_sess];
    stimrate = [stimrate; stimrate_sess];
    stimtype = [stimtype; stimtype_sess];
    
end

