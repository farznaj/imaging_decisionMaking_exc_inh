function [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = alignTraces_prePost_allCases...
    (alignedEvent, traces, traceTimeVec, frameLength, defaultPrePostFrames, shiftTime, scaleTime, ...
    timeInitTone, timeStimOnset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, ...
    trs2rmv, flag_traces, nPreFrames, nPostFrames, onlySetNPrePost, timeStimOffset)

% uses switch case to align traces using the prePost method.
% alignedEvent is one of the follwoing: 
% 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward'

if ~exist('flag_traces', 'var')
    flag_traces = true; % if 1, 'traces' contains the temporal traces for each trial. if 0, it contains the movie for each trial.
end


if exist('nPreFrames', 'var') 
    if isnan(nPreFrames)
        set_npre = 1;
    else
%         nPre0 = nPreFrames;
        set_npre = 0;
    end
else
    set_npre = 1;
end


if exist('nPostFrames', 'var') 
    if isnan(nPostFrames)
        set_npost = 1;
    else
%         nPost0 = nPostFrames;
        set_npost = 0;
    end
else
    set_npost = 1;
end


if ~exist('onlySetNPrePost', 'var') || isempty(onlySetNPrePost)
    onlySetNPrePost = 0;
elseif onlySetNPrePost==1
    traces_aligned_fut = []; time_aligned = []; eventI = [];
end


%%
switch alignedEvent

    %% trial begining (frame 1 of trial when scanning started)
    case 'trialBeg'  
        timeInitTone1 = cellfun(@(x)x(1),timeInitTone);
        eventNow = ones(size(traces)) * frameLength/2;
        eventNow(trs2rmv) = NaN;
        eventBef = [];
        eventAft = timeInitTone1;
        
        eventInds_f = ones(size(traces));
        eventInds_f(trs2rmv) = NaN;
        
        [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        
        if ~onlySetNPrePost
            [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces); % frames x units x trials                
            cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', alignedEvent, nPreFrames, nPostFrames)
        end

        
    %% init tone
    case 'initTone'  
        timeInitTone1 = cellfun(@(x)x(1),timeInitTone);
        eventNow = timeInitTone1;
        eventBef = [];
        eventAft = timeStimOnset;
        
        eventInds_f = eventTimeToIdx(eventNow, traceTimeVec);
        
        if set_npre & set_npost
            [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npre
            [nPreFrames, ~] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npost
            [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
%         else
%             nPreFrames = nPre0;
%             nPostFrames = nPost0;
        end
        
        if ~onlySetNPrePost
            [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces); % frames x units x trials        
            cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', alignedEvent, nPreFrames, nPostFrames)
        end
        
        % if isempty(nPreFrames)
        %     eventI_initTone = size(traces_aligned_fut_initTone,1) - nPostFrames;
        % else
        %     eventI_initTone = nPreFrames+1;
        % end
        
        % if you didn't specify in trs2rmv you need to remove the following from all event times: (outcomes~=1 | correctBut1stUncommitedError | correctBut1stUncommitError)
        %{
        % only analyze correct trials.
        eventInds_f(outcomes~=1) = NaN;
        % exclude trials that entered allow correction or that mouse 1st licked error side.
        eventInds_f(correctBut1stUncommitError | correctButAllowCorr) = NaN;
        sum(~isnan(eventInds_f))

        % since you removed the below from eventTimes, time1stSideTry equals time1stCorrectTry
        % (outcomes~=1 | correctBut1stUncommitedError | correctBut1stUncommitError)
        %}
        
    
    case 'stimOn'    
        %% stim on
        timeInitTone1 = cellfun(@(x)x(1),timeInitTone);
        eventNow = timeStimOnset;
        eventBef = timeInitTone1;
        eventAft = timeCommitCL_CR_Gotone;
        
        eventInds_f = eventTimeToIdx(eventNow, traceTimeVec);
        
%         [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        if set_npre & set_npost
            [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npre
            [nPreFrames, ~] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npost
            [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
%         else
%             nPreFrames = nPre0;
%             nPostFrames = nPost0;
        end
        
        if ~onlySetNPrePost
            [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces); % frames x units x trials        
            cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', alignedEvent, nPreFrames, nPostFrames)
        end
        
        
    case 'stimOff'  
        %% go tone (timeCommitCL_CR_Gotone)
        eventNow = timeStimOffset;
        eventBef = timeStimOnset;
        eventAft = [];        
%         nPreFrames = 3; % 10;

        eventInds_f = eventTimeToIdx(eventNow, traceTimeVec);
        
%         [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        if set_npre & set_npost
            [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npre
            [nPreFrames, ~] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npost
            [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
%         else
%             nPreFrames = nPre0;
%             nPostFrames = nPost0;
        end
        
        if ~onlySetNPrePost
            [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces); % frames x units x trials        
            cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', alignedEvent, nPreFrames, nPostFrames)
        end
        
     
    case 'goTone'  
        %% go tone (timeCommitCL_CR_Gotone)
        eventNow = timeCommitCL_CR_Gotone;
        eventBef = timeStimOnset;
        eventAft = time1stSideTry;        
%         nPreFrames = 3; % 10;

        eventInds_f = eventTimeToIdx(eventNow, traceTimeVec);
        
%         [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        if set_npre & set_npost
            [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npre
            [nPreFrames, ~] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npost
            [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
%         else
%             nPreFrames = nPre0;
%             nPostFrames = nPost0;
        end
        
        if ~onlySetNPrePost
            [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces); % frames x units x trials        
            cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', alignedEvent, nPreFrames, nPostFrames)
        end
        
    
    case '1stSideTry'  
        %% 1st side try
        eventNow = time1stSideTry;
        eventBef = timeCommitCL_CR_Gotone;
        eventAft = timeReward;
        
        eventInds_f = eventTimeToIdx(eventNow, traceTimeVec);
        
%         [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        if set_npre & set_npost
            [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npre
            [nPreFrames, ~] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npost
            [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
%         else
%             nPreFrames = nPre0;
%             nPostFrames = nPost0;
        end
        
        if ~onlySetNPrePost
            [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces); % frames x units x trials        
            cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', alignedEvent, nPreFrames, nPostFrames)
        end
        
    
    case 'reward'  
        %% reward (and commit center lick)
        eventNow = timeReward;
        eventBef = time1stSideTry;
        eventAft = [];
        
        eventInds_f = eventTimeToIdx(eventNow, traceTimeVec);
        
%         [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        if set_npre & set_npost
            [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npre
            [nPreFrames, ~] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npost
            [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
%         else
%             nPreFrames = nPre0;
%             nPostFrames = nPost0;
        end
        
        if ~onlySetNPrePost
            [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces); % frames x units x trials
            cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', alignedEvent, nPreFrames, nPostFrames)
        end
        
        
    case 'commitIncorrResp'
        %% committed incorrect choice
        eventNow = timeCommitIncorrResp;
        eventBef = time1stSideTry;
        eventAft = [];
        
        eventInds_f = eventTimeToIdx(eventNow, traceTimeVec);
        
%         [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        if set_npre & set_npost
            [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npre
            [nPreFrames, ~] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
        elseif set_npost
            [~, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defaultPrePostFrames);
%         else
%             nPreFrames = nPre0;
%             nPostFrames = nPost0;
        end
        
        if ~onlySetNPrePost
            [traces_aligned_fut, time_aligned, eventI, nPreFrames, nPostFrames] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces); % frames x units x trials
            cprintf('blue', 'alignedEvent: %s; nPreFrs= %i; nPostFrs= %i\n', alignedEvent, nPreFrames, nPostFrames)
        end
        
        
        
end


