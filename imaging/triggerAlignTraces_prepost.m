function [traces_aligned_fut, time_aligned, eventI] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime, flag_traces)
% [traces_aligned_fut, time_aligned, eventI] = triggerAlignTraces_prepost(traces, eventInds_f, nPreFrames, nPostFrames, shiftTime, scaleTime)
%
% Align traces on eventInds_f, such that nPreFrames are before the event,
% and nPostFrames are after the event.

% Outputs:
% traces_aligned_fut : traces aligned on the event. frames x units x trials
% eventI : index of the frame that contains the event (on which the traces are aligned).


if ~exist('flag_traces', 'var')
    flag_traces = true; % if 1, 'traces' contains the temporal traces for each trial. if 0, it contains the movie for each trial.
end


%%
if flag_traces % align traces, cell array, each for a trial containing temporal trace of all neurons (frs x units)
    nPreFrames = floor(min([eventInds_f-1, nPreFrames]));
    
    
    framesPerTr = cellfun(@(x) size(x,1), traces);
    nPostFrames = floor(min([framesPerTr(1:length(traces)) - eventInds_f, nPostFrames]));
    
    % traces_aligned = cell(size(traces));
    traces_aligned_fut = NaN(nPreFrames+nPostFrames+1, size(traces{1},2), length(traces));
    
    if nPreFrames < 1,     warning('nPreFrames is 0!'); end
    if nPostFrames < 1,     warning('nPostFrames is 0!'); end
    
    for itr = 1:length(traces)
        if ~isnan(eventInds_f(itr))
            %         trace_preAndEv = traces{itr}(eventInds_f{itr}-nPreFrames : eventInds_f{itr}, :);
            %         trace_postEv = traces{itr}(eventInds_f{itr}+1 : eventInds_f{itr}+nPostFrames, :);
            %
            %         traces_aligned{itr} = [trace_preAndEv, trace_postEv];
            
            %         traces_aligned{itr} = traces{itr}(eventInds_f(itr)-nPreFrames : eventInds_f(itr)+nPostFrames, :);
            traces_aligned_fut(:,:,itr) = traces{itr}(eventInds_f(itr)-nPreFrames : eventInds_f(itr)+nPostFrames, :);
        end
    end
    
    eventI = nPreFrames+1; % index of the frame that contains the event (on which the traces are aligned).
    
    time_aligned = scaleTime * ((1:size(traces_aligned_fut, 1)) - eventI) + shiftTime;
    
    
    
else
    %% align movie_trials, cell array, each for a trial containing its corresponding movie (pix x pix x frs)
%     traces = movie_trials;
    nPreFrames = floor(min([eventInds_f-1, nPreFrames]));
    
    framesPerTr = cellfun(@(x) size(x,3), traces);
    nPostFrames = floor(min([framesPerTr(1:length(traces)) - eventInds_f, nPostFrames]));
    
    % pix x pix x frs x trs
    traces_aligned_fut = NaN(size(traces{1},1), size(traces{1},2), nPreFrames+nPostFrames+1, length(traces));
    
    if nPreFrames < 1,     warning('nPreFrames is 0!'); end
    if nPostFrames < 1,     warning('nPostFrames is 0!'); end
    
    for itr = 1:length(traces)
        if ~isnan(eventInds_f(itr))
            traces_aligned_fut(:,:,:,itr) = traces{itr}(:,:, eventInds_f(itr)-nPreFrames : eventInds_f(itr)+nPostFrames); % pix x pix x frs x trs
        end
    end
    
    eventI = nPreFrames+1; % index of the frame that contains the event (on which the traces are aligned).
    
    time_aligned = scaleTime * ((1:size(traces_aligned_fut, 1)) - eventI) + shiftTime;
    
end

