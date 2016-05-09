function [trs2rmv, stimdur, stimrate, stimtype, cb] = setTrs2rmv_final(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, imagingFlg, badAlignTrStartCode, trialStartMissing, trialCodeMissing)
% the complete function for setting trs2rmv, it calls setTrs2rmv

if ~exist('imagingFlg', 'var')
    imagingFlg = 0; % when analysis only behavioral data.
end


%%
cb = unique([alldata.categoryBoundaryHz]); % category boundary in hz


% set the following as trs2rmv: trs_begWarmUp(:); trs_helpedInit(:); trs_helpedChoice(:), trs_extraStim, trs_shortWaitdur, trs_problemAlign, trs_badMotion_pmtOff
if imagingFlg
    trs2rmv = setTrs2rmv(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs, badAlignTrStartCode, trialStartMissing, trialCodeMissing);
else
    trs2rmv = setTrs2rmv(alldata, thbeg, excludeExtraStim, excludeShortWaitDur, begTrs);
end

%%%%%%% Set stim rate: trials with stimrate=nan, must be added to trs2rmv as well
[~, stimdur, stimrate, stimtype] = setStimRateType(alldata); % stimtype = [multisens, onlyvis, onlyaud];
if sum(isnan(stimrate)) > 0
    fprintf('# trials with NaN stimrate= %i\n', sum(isnan(stimrate)))
    trs2rmv = unique([trs2rmv; find(isnan(stimrate))]);
end

%{
trs2rmv = unique([trs2rmv, trs_unwantedOutcome]);

% strictly correct trials: no allow correction, no error lick (even if not committed).
trs2rmv = unique([trs2rmv, trs_allowCorrectEntered, trs_errorlick_again_wait_entered]);
%}


%%%%%%% take care of cases that go tone happened earlier than stim offset
%%%%%%% and resulted in a different stim type.
fractFlashBefGoTone = NaN(1, length(alldata));
stimRateChanged = false(1, length(alldata));
goToneRelStimOnset = NaN(1, length(alldata));

for tr = 1:length(alldata)
    if ~isempty(alldata(tr).parsedEvents.states.center_reward) % ~isnan(goToneRelStimOnset(tr))
        % xx shows flashes/clicks onset time during the stimulus
        xx = cumsum(alldata(tr).auditoryIeis + alldata(tr).eventDuration);
        xx = [0 xx(1:end-1)] * 1000 + 1; % ms % 1st flash onset is assigned 1
        
        
        % goToneRelStimOnset = timeCommitCL_CR_Gotone - timeStimOnset + 1;
        goToneRelStimOnset(tr) = ((alldata(tr).parsedEvents.states.center_reward(1))*1000 - (alldata(tr).parsedEvents.states.wait_stim(1))*1000) + 1;
        
        
        % percentage of flashes/clicks that happened before go tone. If 1,
        % the animal received the whole stimulus before go tone.
        fractFlashBefGoTone(tr) = sum(xx < goToneRelStimOnset(tr)) / length(xx);
        
        % did stim type (hr vs lr) at go tone change compared to the
        % original stimulus?
        if sign(length(xx)-cb) ~= sign((sum(xx < goToneRelStimOnset(tr)))-cb) % alldata(tr).nAuditoryEvents
            stimRateChanged(tr) = true;
        end
    end
end

% find(fractFlashBefGoTone<1) % in these trials go tone came before the entire flashes were played.
% fractFlashBefGoTone_ifLessThan1 = fractFlashBefGoTone(fractFlashBefGoTone<1);
% fprintf('Fract flash before goTone if <1\n%.2f \n', fractFlashBefGoTone(fractFlashBefGoTone<1))
if sum(fractFlashBefGoTone<1)>0
    %     nanmean(fractFlashBefGoTone(fractFlashBefGoTone<1))
    aveFractFlashBefGoTone_forLessThan1 = nanmean(fractFlashBefGoTone(fractFlashBefGoTone<1));
    fprintf('%.2f= mean fraction of flashes occurred before goTone in trials with goTone earier than stimOffset\n', aveFractFlashBefGoTone_forLessThan1)
    
    figure; plot(fractFlashBefGoTone)
    xlabel('Trial')
    ylabel([{'Fraction of flashes played before Go Tone'}, {'Ideally we want 1.'}])
end


if any(stimRateChanged)
    trsStimTypeScrewed = find(stimRateChanged)'; % in these trials when go tone came stim type (hr vs lr) was different from the entire stim... these are obviously problematic trials.
    fprintf('%d = #trs with a different stim type at Go Tone than the actual stim type\n', length(trsStimTypeScrewed))
    aveFractFlashBefGoTone_stimRateChanged = nanmean(fractFlashBefGoTone(stimRateChanged));
    fprintf('%.2f= mean fraction of flashes occurred before goTone in trials that stimType changed dur to goTone earier bein than stimOffset\n', aveFractFlashBefGoTone_stimRateChanged)
    
    aveOutcomeFractFlashBefGoTone_stimRateChanged = nanmean([alldata(stimRateChanged).outcome]);
    fprintf('%.2f= mean outcome on trials with stimType Changed\n', aveOutcomeFractFlashBefGoTone_stimRateChanged)
    
    trs2rmv = unique([trs2rmv; trsStimTypeScrewed]);
end

fprintf('Number of trs2rmv: %d\n', length(trs2rmv))
disp(trs2rmv')

