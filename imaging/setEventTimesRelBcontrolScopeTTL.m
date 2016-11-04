function [timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks, timeStimOnsetAll] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime, stimAftGoToneParams, outcomes)
% Set the time of events that happen during each trial.
%
% By default all times (ms) are relative to when bcontrol sent scope ttl. 
% unless scopeTTLOrigTime is set to false, in which case all times will be
% relative to the begining of the trials in bcontrol.

if ~exist('scopeTTLOrigTime', 'var')
    scopeTTLOrigTime = true;
end

if ~exist('stimAftGoToneParams', 'var') % this is to take care of trials for which stimulus continued after the go tone.
    stimAftGoToneParams = [];
    
elseif ~isempty(stimAftGoToneParams)
    frameLength = 1000/30.9; % sec.
    rmv_timeGoTone_if_stimOffset_aft_goTone = stimAftGoToneParams{1};
    rmv_time1stSide_if_stimOffset_aft_1stSide = stimAftGoToneParams{2};
    setNaN_goToneEarlierThanStimOffset = stimAftGoToneParams{3};
    % stimAftGoToneParams = {rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, setNaN_goToneEarlierThanStimOffset};    
end

% remember if you change outcomes for allowCorrEntered trials, then time of
% events will reflect that, bc outcomes is provided as input.

%%
% if you define outcomes as below, you wont correct for trials that mouse
% did allow correction.
% outcomes = [alldata.outcome]; 
%{
% set to failure the outcome of trials on which the mouse used allow correction (bc we want to consider the outcome of the first choice)
a = arrayfun(@(x)x.parsedEvents.states.punish_allowcorrection, alldata, 'uniformoutput', 0);
outcomes(~cellfun(@isempty, a)) = 0;
%}

invalid = [-1 -3 -4];
noDecision = -2; % mouse did not lick side at all
noSideCommit = -5; % mouse did not commit his side lick.
% noFinalChoice = [-2 -5];

timeNoCentLickOnset = NaN(size(alldata));
timeNoCentLickOffset = NaN(size(alldata));
timeInitTone = cell(size(alldata)); timeInitTone(:) = {NaN};
time1stCenterLick = NaN(size(alldata));
timeStimOnsetAll = NaN(size(alldata));
timeStimOffset = NaN(size(alldata));
timeCommitCL_CR_Gotone = NaN(size(alldata));
time1stSideTry = NaN(size(alldata));
time1stCorrectTry = NaN(size(alldata));
time1stIncorrectTry = NaN(size(alldata));
timeReward = NaN(size(alldata));
timeCommitIncorrResp = NaN(size(alldata));
time1stCorrectResponse = NaN(size(alldata));
timeStop = NaN(size(alldata));
centerLicks = cell(size(alldata)); centerLicks(:) = {NaN};
leftLicks = cell(size(alldata)); leftLicks(:) = {NaN};
rightLicks = cell(size(alldata)); rightLicks(:) = {NaN};


%% By default: all times (ms) are relative to when bcontrol sent scope ttl. (this makes more sense than timeStartScan_abs, since alldata.frameTimes is also relative to bcontrol ttl time.)
% unless scopeTTLOrigTime is set to false, in which case all times will be
% relative to the begining of the trials in bcontrol.

for tr = 1:length(alldata)
    
    
    if ~ismember(tr, trs2rmv) & ~isnan(outcomes(tr))
        
        % set t0 : event times will be relative to t0.
        
        if scopeTTLOrigTime
            if isfield(alldata(tr).parsedEvents.states, 'trial_start_rot_scope')
                timeScopeTTLsent_abs = alldata(tr).parsedEvents.states.trial_start_rot_scope(1,1) * 1000; % _abs indicates that the time is relative to the beginning of the session (not the begining of the trial).
            else
                timeScopeTTLsent_abs = alldata(tr).parsedEvents.states.start_rotary_scope(1,1) * 1000;
            end
            t0 = timeScopeTTLsent_abs; % by default event times will be relative to when bcontorl sent scopeTTL, this will be useful for analyzing imaging data in line with behavioral data. 
            
        else
            t0 = alldata(tr).parsedEvents.states.state_0(2) * 1000; % the beginning of a trial in bcontrol. This will be useful when looking at behavioral data independent of imaging data.
        end
        
        % duration (ms) that MScan lagged behind in executing bcontrol command signal
        %         mscanLagRelBcontrol = alldata(tr).frameTimes(1) - frameLength/2;
        
        % this is what you need to subtract from the timepoints in a trial to find
        % how far they are from the actual scanning origin (instead of when bcontrol sent the scope TTL).
        %         timeStartScan_abs = timeScopeTTLsent_abs + mscanLagRelBcontrol; % in ms (time 0 is beginning of the session).
        
        
        
        %% time of no_centerLick onset and offset (they happen during start_rotary_scope)
        
        last_start_rotary_scope = alldata(tr).parsedEvents.states.start_rotary_scope(end,:);
        if ~isempty(alldata(tr).parsedEvents.states.start_rotary_scope2)
            last_start_rotary_scope = [last_start_rotary_scope ; alldata(tr).parsedEvents.states.start_rotary_scope2(end,:)]; 
        end
        f = ismember(last_start_rotary_scope(:,2), alldata(tr).parsedEvents.states.trialcode1(1));
        timeNoCentLickOnset(tr) = last_start_rotary_scope(f,1) * 1000 - t0; % onset of the epoch during which mouse did not lick center (and side)
        timeNoCentLickOffset(tr) = last_start_rotary_scope(f,2) * 1000 - t0; % same as trialCode onset. % offset of the epoch during which mouse did not lick center (and side)
        
        
        %% times of trial initiation tones. % equals to end of trialCode state
        
        timeInitTone{tr} = [alldata(tr).parsedEvents.states.wait_for_initiation(:,1);...
            alldata(tr).parsedEvents.states.wait_for_initiation2(:,1)] * 1000 - t0;
        
        
        %% time of 1st lick (when trial was correctly initiated), relative to scan start
        
        if ~ismember(outcomes(tr), -3) % dont analyze wrong start.
            time1stCenterLick(tr) = max([alldata(tr).parsedEvents.states.wait_for_initiation(:,2);...
                alldata(tr).parsedEvents.states.wait_for_initiation2(:,2)]) * 1000 - t0;
        end
        
        
        %% stimulus onset. (it happens after stim_delay since the initi time).
        
        %         if ~ismember(outcomes(tr), -3) % dont analyze wrong start (change -3 to invalid if you don't want to analyze early decision or no center commit for the stimOnset event)
        if ~isempty(alldata(tr).parsedEvents.states.wait_stim) %&& ~ismember(outcomes(tr), -1) % -1 added so early decisions are not analyzed for stim, bc the stim ends at the middle for them. % you commented the line above bc if the mouse licks during stim pre delay it will be counted as early decision (-1) hence outcome is not -3 but still no stimulus was played.
            timeStimOnsetAll(tr) = alldata(tr).parsedEvents.states.wait_stim(1) * 1000 - t0; % includes outcome=-1 trials.
        end
        
        
        %% commit center lick , center reward , go tone, relative to scan start.
        
        if ~ismember(outcomes(tr), invalid) % dont analyze early decision , wrong start , or no center commit
            timeCommitCL_CR_Gotone(tr) = alldata(tr).parsedEvents.states.center_reward(1) * 1000 - t0;
        end
        
        
        %% first 1st side try and 1st correct/ incorrect try, relative to scan start. 
        % calling "try" instead of response because mouse did not necesarily
        % confirm these licks (ie the 2nd lick).
        
        if ~ismember(outcomes(tr), [invalid, noDecision])
            if ~isempty(alldata(tr).parsedEvents.states.correctlick_again_wait)
                time1stCorrectTry(tr) = min(alldata(tr).parsedEvents.states.correctlick_again_wait(:,1)) * 1000 - t0;
            end
            
            if ~isempty(alldata(tr).parsedEvents.states.errorlick_again_wait)
                time1stIncorrectTry(tr) = min(alldata(tr).parsedEvents.states.errorlick_again_wait(:,1)) * 1000 - t0;
            end
            
            time1stSideTry(tr) = min([alldata(tr).parsedEvents.states.correctlick_again_wait(:,1);...
                alldata(tr).parsedEvents.states.errorlick_again_wait(:,1)]) * 1000 - t0;
        end
        
        
        %% commit correct lick , reward, relative to scan start.
        
        % note: some of the trials included here are allow-correction
        % trials, ie the mouse first chose the wrong side and then
        % corrected for it.
        if ismember(outcomes(tr), 1)
            timeReward(tr) = alldata(tr).parsedEvents.states.reward(1) * 1000 - t0;
        end
        
        
        %% commit (lick again) incorrect lick, relative to scan start.
        % note: in some of these trials, mouse eventually corrected his
        % choice.
        
        if ~ismember(outcomes(tr), [invalid, noSideCommit]) % remember if outcomes is defined as [alldata.outcome], it means mouse may commit a wrong choice and then switch to a correct one and not commit it, and it will be counted as noSideCommit.
            if ~isempty(alldata(tr).parsedEvents.states.punish_allowcorrection)
                % Mouse may enter punish_allowcorrection more than once :
                % He enters it and then exits it by doing a correct lick.
                % But he does not commit the correct lick, instad he does
                % an error lick and commit it, so he re-enters the
                % punish_allowcorrection. In this case there is more than 1
                % commitIncorrectResp... we go with the 1st one... but keep
                % it in mind.
                ne = size(alldata(tr).parsedEvents.states.punish_allowcorrection, 1);
                if ne > 1
                    warning(sprintf('Mouse entered punish_allowcorrect %i times. We pick the 1st one as commitIncorrResp\n', ne))
                end
                timeCommitIncorrResp(tr) = alldata(tr).parsedEvents.states.punish_allowcorrection(1,1) * 1000 - t0;
            elseif ~isempty(alldata(tr).parsedEvents.states.punish)
                timeCommitIncorrResp(tr) = alldata(tr).parsedEvents.states.punish(:,1) * 1000 - t0;
            end
        end
        
        
        %% stop_rotary_scope
        
        timeStop(tr) = alldata(tr).parsedEvents.states.stop_rotary_scope(1,1) * 1000 - t0;
        
        
        %% time of all licks : negative values indicate licks that happened before the start of scopeTTL.
        
        centerLicks{tr} = alldata(tr).parsedEvents.pokes.C(:,1) * 1000 - t0;
        leftLicks{tr} = alldata(tr).parsedEvents.pokes.L(:,1) * 1000 - t0;
        rightLicks{tr} = alldata(tr).parsedEvents.pokes.R(:,1) * 1000 - t0;
        
        
        %% stimulus offset
        
        if ~ismember(outcomes(tr), -3)            
            % end of the stim if stim was not aborted due to mouse going to one of the states written below.
            stimEndIfNoAbort = (alldata(tr).parsedEvents.states.wait_stim(1) * 1000) + (alldata(tr).stimDuration * 1000);
            
            % states that will terminate the stimulus.
            if ~isempty(alldata(tr).parsedEvents.states.early_decision0)
                stimAbortTime = alldata(tr).parsedEvents.states.early_decision0(1) * 1000;
                
            elseif ~isempty(alldata(tr).parsedEvents.states.did_not_lickagain)
                stimAbortTime = alldata(tr).parsedEvents.states.did_not_lickagain(1) * 1000;
                
            elseif ~isempty(alldata(tr).parsedEvents.states.did_not_choose)
                stimAbortTime = alldata(tr).parsedEvents.states.did_not_choose(1) * 1000;
                
            elseif ~isempty(alldata(tr).parsedEvents.states.did_not_sidelickagain)
                stimAbortTime = alldata(tr).parsedEvents.states.did_not_sidelickagain(1) * 1000;
                
            elseif ~isempty(alldata(tr).parsedEvents.states.direct_correct)
                stimAbortTime = alldata(tr).parsedEvents.states.direct_correct(1) * 1000;
                
            elseif ~isempty(alldata(tr).parsedEvents.states.reward_stopstim)
                stimAbortTime = alldata(tr).parsedEvents.states.reward_stopstim(1) * 1000;
                
            elseif ~isempty(alldata(tr).parsedEvents.states.punish_allowcorrection_done)
                stimAbortTime = alldata(tr).parsedEvents.states.punish_allowcorrection_done(1) * 1000;
                
            elseif ~isempty(alldata(tr).parsedEvents.states.punish)
                stimAbortTime = alldata(tr).parsedEvents.states.punish(1) * 1000;
                
            end
            % stimEndIfNoAbort, stimAbortTime
            
            stimEndActual = min(stimEndIfNoAbort, stimAbortTime);
            timeStimOffset(tr) = stimEndActual - t0;            
        end
%         % stimulus offset
%         if ~ismember(outcomes(tr), -3)
%             timeStimOffset(tr) = alldata(tr).parsedEvents.states.wait_stim(2) * 1000 - t0;
%         end

        
        %% first correct lick (not the 2nd, commit lick) that resulted in a reward. It is a useful var in case mouse made side licks without committing them: time1stCorrectResponse assures that this lick try was followed by a commit lick (unlike time1stCorrectTry).
        
        if ismember(outcomes(tr), 1)
            caw = alldata(tr).parsedEvents.states.correctlick_again_wait;
            ca = alldata(tr).parsedEvents.states.correctlick_again;
            r = alldata(tr).parsedEvents.states.reward(1);
            caw_yes = caw(ismember(ca(:,2),r),1); % first correct lick that resulted in a reward.
            
            time1stCorrectResponse(tr) = caw_yes * 1000 - t0;
            
            % if the correct response was followed in shorter than 100ms by
            % a center lick, exclude that trial.
            %{
            % check
            centerLick = alldata(tr).parsedEvents.pokes.C(:,1);
            leftLick = alldata(tr).parsedEvents.pokes.L(:,1);
            rightLick = alldata(tr).parsedEvents.pokes.R(:,1);
            
            firstResp = min([alldata(tr).parsedEvents.states.correctlick_again_wait(:,1);...
                alldata(tr).parsedEvents.states.errorlick_again_wait(:,1)]);
            
            % list of licks after the response lick.
            centerLick(centerLick < firstResp) = [];
            leftLick(leftLick < firstResp) = [];
            rightLick(rightLick < firstResp) = [];
            
            cf = centerLick(centerLick>=caw_yes);
            lf = leftLick(leftLick>=caw_yes);
            rf = rightLick(rightLick >= caw_yes);
            if ~isempty(rf) && ~isempty(lf)
                fprintf('trial %d: mouse licked on the two sides after making a correct response.\n', tr)
            end
            
            % if the correct response was followed in shorter than 100ms by
            % a center lick, exclude that trial.
            if max(time1stCorrectResponse(tr) - (cf * 1000 - t0)) >= -100
                time1stCorrectResponse(tr) = NaN;
            end

            %}
            %             disp('------')
            %             pause
            
        end
        
    end
    
end

timeStimOnset = timeStimOnsetAll; % like timeStimOnsetAll except it does not include outcome=-1 trials
timeStimOnset(ismember(outcomes, -1)) = NaN; % exclude outcome=-1 trials


%%

trsGoToneEarlierThanStimOffset = find(timeCommitCL_CR_Gotone < timeStimOffset)';


%% Take care of trials that mouse received center reward and go tone before the
% end of the stimulus. This is important if looking at neural responses
% during stim interval: you dont want to look beyond goTone time for these
% trials... not necessarily important for other alignments. So you may or
% may not need to run this part! If want to be conservative, run it!

% but also remember when you align traces on 1stSideTry you don't want to
% have stimulus played during the baseline. Usually 1stSideTry happens
% long after stimOffset so this shouldn't be a problem. Figures below help
% you dig into this. If you want to be very careful, you should remove
% trials that have stim during n frames before time1stSideTry where n =
% nPreFrame for alignments on 1stSideTry. Below you are removing trials
% with stimOffset during the 3 frames before 1stSideTry.

% rmv_timeGoTone_if_stimOffset_aft_goTone = 0; % if 1, trials with stimOffset after goTone will be removed from timeGoTone (ie any analyses that aligns trials on the go tone)
% rmv_time1stSide_if_stimOffset_aft_1stSide = 0; % if 1, trials with stimOffset after 1stSideTry will be removed from time1stSideTry (ie any analyses that aligns trials on the 1stSideTry)

if isempty(trsGoToneEarlierThanStimOffset)
    fprintf('No trials with goTone earlier than stimOffset :)\n')

elseif ~isempty(stimAftGoToneParams)
    
    figure('name', 'Trials with goTone earlier than stimulus offset');
    subplot(211), hold on
    plot(timeStimOffset(trsGoToneEarlierThanStimOffset)  -  ...
        timeCommitCL_CR_Gotone(trsGoToneEarlierThanStimOffset))
    
    plot(time1stSideTry(trsGoToneEarlierThanStimOffset)  -  ...
        timeStimOffset(trsGoToneEarlierThanStimOffset))
    plot([1 length(trsGoToneEarlierThanStimOffset)], [3*frameLength 3*frameLength], 'k:')
    ylabel('Time (ms)')
    legend('post-goTone duration with stimulus', 'pre-1stSideTry duration without stimulus') % this is a more understandable legend
    %     legend('stimOffset - goTone', '1stSideTry - stimOffset')
    
    subplot(212), hold on;
    plot(timeCommitCL_CR_Gotone(trsGoToneEarlierThanStimOffset))
    plot(timeStimOffset(trsGoToneEarlierThanStimOffset), 'r')
    plot(time1stSideTry(trsGoToneEarlierThanStimOffset), 'g')
    plot(timeReward(trsGoToneEarlierThanStimOffset), 'm')
    plot(timeCommitIncorrResp(trsGoToneEarlierThanStimOffset), 'c')
    xlabel('Trials with Go tone earlier than stim offset')
    ylabel('Time (ms)')
    legend('goTone', 'stimOffset', '1stSideTry', 'Reward', 'commitIncorr')
    
    
    %%
    cprintf('blue', '%d trials with goTone earlier than stimOffset...\n', length(trsGoToneEarlierThanStimOffset))
    if rmv_timeGoTone_if_stimOffset_aft_goTone
        cprintf('blue', '... removing them from timeCommitCL_CR_Gotone!\n')
        % timeStimOnset(trsGoToneEarlierThanStimOffset) = NaN; % IMPORTANT: you dont want to look beyond goTone time for these trials, but upto go tone is safe!
        timeCommitCL_CR_Gotone(trsGoToneEarlierThanStimOffset) = NaN;  % you want this for sure (bc in some trials stim has been still playing after go tone).
    else
        a0 = length(trsGoToneEarlierThanStimOffset) / length(timeCommitCL_CR_Gotone);
        cprintf('blue','... you chose not to remove them. In %.2f trials stimulus continues after the goTone!\n', a0)
    end
    
    
    %%
    a = time1stSideTry(trsGoToneEarlierThanStimOffset) - timeStimOffset(trsGoToneEarlierThanStimOffset);
    if sum(a < 3*frameLength)
        cprintf('blue','%i trials with stimOffset within 3 frames before 1stSideTry.\n', sum(a < 3*frameLength))
        
        if rmv_time1stSide_if_stimOffset_aft_1stSide
            fprintf('... removing them from 1stSideTry, 1stCorrTry, 1stIncorrTry!\n')
            aa = trsGoToneEarlierThanStimOffset(a < 3*frameLength);
            time1stSideTry(aa) = NaN;
            time1stCorrectTry(aa) = NaN;
            time1stIncorrectTry(aa) = NaN;
            time1stCorrectResponse(aa) = NaN;
        else
            a0 = sum(a < 3*frameLength) / length(time1stSideTry);
            warning('... you chose not to remove them. In %.2f trials stimulus continues until 1stSideTry!\n', a0)
        end
    else
        cprintf('blue','No trials with stimOffset within 3 frames before 1stSideTry. Good :) \n')
    end
    
    
    %%
    % For the following events not as important to set to nan unless you think
    % neural responses aligned on these events can be different when go tone
    % happened during stim, which is likely!
    if setNaN_goToneEarlierThanStimOffset % if 1, set to nan eventTimes of trials that had go tone earlier than stim offset... if 0, only goTone time will be set to nan.
        timeStimOnset(trsGoToneEarlierThanStimOffset) = NaN; % you dont want to look beyond goTone time for these trials, but upto go tone is safe!
        timeStimOffset(trsGoToneEarlierThanStimOffset) = NaN;
        time1stSideTry(trsGoToneEarlierThanStimOffset) = NaN;
        time1stCorrectTry(trsGoToneEarlierThanStimOffset) = NaN;
        time1stIncorrectTry(trsGoToneEarlierThanStimOffset) = NaN;
        timeReward(trsGoToneEarlierThanStimOffset) = NaN;
        timeCommitIncorrResp(trsGoToneEarlierThanStimOffset) = NaN;
        time1stCorrectResponse(trsGoToneEarlierThanStimOffset) = NaN;
    end
    
    
end



% if you want to be super conservative, just reset trs2rmv and again set the times:
% trs2rmv = unique([trs2rmv; trsGoToneEarlierThanStimOffset]);
% [timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
%     time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
%     setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv);




%%% Done: Important: take care of the issue that mouse may have made his decision before the end of the stimulus.
% make sure when looking at the stimulus epoch, it is all
% stimulus and does not include choice or episodes of stimulus absence!
%
% timeStimOffset in setEventTimesRelBcontrolScopeTTL.m accurately shows
% stim offset.
% but mouse may have committed center lick (hence received cent reward and
% go tone) before stim offset.
% timeCommitCL_CR_Gotone shows the time of commit cent lick and go tone, so
% if timeCommitCL_CR_Gotone < timeStimOffset, go tone (and cent reward)
% happened before the stim was over, so you need to exclude these trials.
%
% this is bc not only stim interval includes cent reward and go tone, but
% also stim rate might be different at the time of go tone than what the
% final rate which determins the reward.
%
% [(1:length(timeCommitCL_CR_Gotone))' timeCommitCL_CR_Gotone' , timeStimOffset' , [timeCommitCL_CR_Gotone < timeStimOffset]']

%{
% this plot shows when flashes/clicks happened (their onset, remember they
% last for alldata.eventDuration) during the stimulus
yy = ones(1, length(alldata(tr).auditoryIeis));
figure; plot(xx, yy, 'o')

% Not useful, what matters is when the go tone was played not when waitdur ended: how many events were played (ie their onset had happened) when the waitDur happened.
% sum(xx < alldata(tr).waitDuration * 1000)

diff([[alldata.waitDuration]', [alldata.stimDuration]'], [], 2)

% trials with different stimdur_diff than waitdur
if any([alldata.stimDur_diff])
    [alldata.stimDur_diff] ~= [alldata.waitDuration]
end
%}



