function [left, right] = Behavior_getStimEvent(stimType, stimEvents)
% short code to isolate stimulus event, depending on stimulus type.
% returns absolute numbers of events on left/right side.

%% identify stimtype and assign channels
if stimType == 1 %vision
    chInd = [3 4];
elseif stimType == 2 %audio
    chInd = [1 2];
elseif stimType == 4 %somatosensory
    chInd = [5 6];
elseif stimType == 3 %audiovisual
    chInd = [1 3; 2 4]';
elseif stimType == 5 %somatovisual
    chInd = [2 5; 4 6]';
elseif stimType == 6 %somatoaudio
    chInd = [1 5; 2 6]';
elseif stimType == 5 %all mixed
    chInd = [1 3 5;2 4 6]';
else
    error('unknown stimtype');
end

%% identify left and rightward events
left = stimEvents(chInd(:,1));
left = cat(2,left{:});

right = stimEvents(chInd(chInd ~= chInd(:,1)));
right = cat(2,right{:});

end

