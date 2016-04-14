%{
EditParam(obj, 'maxPastBad', 3, xpos, ypos); next_row(ypos); % maximum number of past trials with "bad" outcome, in order to make waitDur short.
SoloParamHandle(obj, 'waitDurOrig', 'value', NaN);
SoloParamHandle(obj, 'waitDurStepOrig', 'value', NaN);

SoloParamHandle(obj, 'sideLickDurOrig', 'value', NaN);
SoloParamHandle(obj, 'sideLickDurStepOrig', 'value', NaN);

badOutcomes = [-3, -1, -4, 2]; % wrongInitiation, earlyDecision, didNotLickAgain, didNotChoose % add 0 if you want to also include punishment.
pastBad = ismember(HitHistory(n_done_trials - value(maxPastBad) : n_done_trials-1), badOutcomes);
%}


%%

function [waitdo, sideldo, waitd, waitds, sideld, sidelds] = adaptiveDurs(pastBad, value(maxPastBad), ...
    value(waitDurOrig), value(sideLickDurOrig), value(WaitDuration), value(WaitDurationStep), value(SideLickDur), value(SideLickDurStep))

% waitDur and sideLickDur: reset their values if there are
% maxPastBad consecutive bad trials, otherwise keep using the GUI values.
if sum(pastBad) == value(maxPastBad)
    % save the current duration and step values for access in future.
    waitDurOrig.value = value(WaitDuration);  
    waitDurStepOrig.value = value(WaitDurationStep);

    sideLickDurOrig.value = value(SideLickDur);  
    sideLickDurStepOrig.value = value(SideLickDurStep);

    % rest (considerably lower) the duration values but increase step values.
    WaitDuration.value = .15; % WaitDuration.value - .25;    
    SideLickDur.value = value(SideLickDur) - .05;
end



% waitDurStep: gradually update as waitDur gets close to the
% original value (ie the value before change), when there has been consecutive bad trials.
if ~isnan(value(waitDurOrig)) % there has never been consecutive bad trials, or if there has been some in the past, the animal has already gone back to its original value of waitDur.
    if abs(value(WaitDuration) - value(waitDurOrig)) <= .005 % use the original (before resetting) step values and set orig values to NaN, so later on the GUI values for the step will be used.
        WaitDurationStep.value = value(waitDurStepOrig);
        waitDurOrig.value = NaN;

    elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .01
        WaitDurationStep.value = .0005;

    elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .02
        WaitDurationStep.value = .001;

    elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .1
        WaitDurationStep.value = .005;
        
    else
        WaitDurationStep.value = .01;

    end
end


% sideLickDurStep: gradually update as sideLickDur gets close to the
% original value (ie the value before change), when there has been consecutive bad trials.
if ~isnan(value(sideLickDurOrig))  % there has never been consecutive bad trials, or if there has been some in the past, the animal has already gone back to its original value of sideLickDur.
    if abs(value(SideLickDur) - value(sideLickDurOrig)) <= .002 % use the original (before resetting) step values and set orig values to NaN, so later on the GUI values for the step will be used.
        SideLickDurStep.value = value(sideLickDurStepOrig);
        sideLickDurOrig.value = NaN;
        
    elseif abs(value(SideLickDur) - value(sideLickDurOrig)) <= .005
        SideLickDurStep.value = .0005;

    elseif abs(value(SideLickDur) - value(sideLickDurOrig)) <= .01
        SideLickDurStep.value = .001;

    elseif abs(value(SideLickDur) - value(sideLickDurOrig)) <= .02
        SideLickDurStep.value = .0015;
        
    else
        SideLickDurStep.value = .003;

    end
end


waitdo = value(waitDurOrig); 
sideldo = value(sideLickDurOrig);
waitd = value(WaitDuration);
waitds = value(WaitDurationStep);
sideld = value(SideLickDur);
sidelds = value(SideLickDurStep);


end




%%


SoloParamHandle(obj, 'durs_adaptive', 'value', 1);





