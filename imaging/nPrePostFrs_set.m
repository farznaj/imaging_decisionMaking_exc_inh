function [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defPrePostFrs)
% [nPreFrames, nPostFrames] = nPrePostFrs_set(eventNow, eventBef, eventAft, frameLength, defPrePostFrs);
%
% set number of frames preceding (nPreFrames) and following (nPostFrames)
% the event of interest, which will be used by triggerAlignTraces_prepost to set trigger-aligned traces.
% nPreFrames and nPostFrames are the number of frames between eventNow,
% eventBef and eventNow, eventAft.


%%

if ~exist('defPrePostFrs', 'var')
    defPrePostFrs = 2;
end

ntrs = sum(~isnan(eventNow));
fprintf('Number of trials: %d\n', ntrs)

if ~isempty(eventBef)
    nPreTime = min(eventNow - eventBef);
    nPreFrames = floor(nPreTime/frameLength);
    
    if nPreFrames<1
        nPreFrames = defPrePostFrs;
        
        a = sum((eventNow - eventBef) < nPreFrames*frameLength); % in these trials during the baseline (nPreFrames) of traces_aligned_fut, eventBef happened.
        disp('Using default value for nPreFrames!')
        fprintf('%.2f: Fraction of trials with eventBef happenning during baseline (nPreFrames) of eventNow\n', a/ntrs)
        % in these trials stimon was preceded by init tone in <1 frame.
        % so in these trials you may see a rise right before the stim due to
        % init... or there will be confound in init tone-evoked and stim-evoked responses.
    end
else
    nPreFrames = [];
end


if ~isempty(eventAft)
    nPostTime = min(eventAft - eventNow);
    nPostFrames = floor(nPostTime/frameLength);
    
    if nPostFrames<1
        nPostFrames = defPrePostFrs;
        
        a = sum((eventAft - eventNow) < nPostFrames*frameLength); % in these trials during the baseline (nPreFrames) of traces_aligned_fut, eventBef happened.
        disp('Using default value for nPostFrames!')
        fprintf('%.2f: Fraction of trials with eventAft happenning during nPostFrames\n', a/ntrs)
        % in these trials stimon was preceded by init tone in <1 frame.
        % so in these trials you may see a rise right before the stim due to
        % init... or there will be confound in init tone-evoked and stim-evoked responses.
    end    
else
    nPostFrames = [];
end




