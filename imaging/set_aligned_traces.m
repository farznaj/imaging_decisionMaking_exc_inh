% Set event-aligned traces. Extremely careful thought is taken in what trials will be
% included for each alighment:

% Run imaging_prep_analysis to get the required vars.

% Trials that will go into each of the event-aligned traces:
% stimOnset alignment: trials with stimulus duration >= th_stim_dur and go tone after ep_ms(2)
% goTone alignment: both sets of traces with potential stim after go tone and making sure no stim after go tone.
% 1stSideTry: both sets of traces for no COM trials (1stSideTry is for sure followed by a commit lick on that same side) and COM trials (try without commit).
% reward: all trials 
% incorrResp: all trials (incorr resp followed by reward due to allowCorr can be excluded if desired). 

save_aligned_traces = 1; % if 1, the following structures will be appended to postName: 'stimAl', 'firstSideTryAl', 'firstSideTryAl_COM', 'goToneAl', 'goToneAl_noStimAft', 'rewardAl', 'commitIncorrAl'


%%
[pd, pnev_n] = fileparts(pnevFileName);
postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));


traces = alldataSpikesGood; % alldataSpikesGoodExc; % alldataSpikesGoodInh; % alldataSpikesGood;  % traces to be aligned.
dofilter = false;
traceTimeVec = {alldata.frameTimes}; % time vector of the trace that you want to realign.
% set nPre and nPost to nan if you want to go with the numbers that are based on eventBef and eventAft.
% set to [] to include all frames before and after the alignedEvent.
nPreFrames = []; % nan;
nPostFrames = []; % nan;

% good_inhibit = inhibitRois==1;
% good_excit = inhibitRois==0;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set outcomes

allowCorrectResp = 'change'; % 'change'; 'remove'; 'nothing'; % if 'change': on trials that mouse corrected his choice, go with the original response.
uncommittedResp = 'nothing'; % 'change'; 'remove'; 'nothing'; % what to do on trials that mouse made a response (licked the side port) but did not lick again to commit it.

[outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp);

% set trs2rmv to nan
outcomes(trs2rmv) = NaN;
allResp(trs2rmv) = NaN;
allResp_HR_LR(trs2rmv) = NaN;


%% Set event times (ms) relative to when bcontrol starts sending the scope TTL. event times will be set to NaN for trs2rmv.

% remember if you change outcomes for allowCorrEntered trials, then time of
% events will reflect that.

% setNaN_goToneEarlierThanStimOffset = 0; % if 1, set to nan eventTimes of trials that had go tone earlier than stim offset... if 0, only goTone time will be set to nan.
scopeTTLOrigTime = 1;

% I set all to 0 bc I want to get all times. but later I will take care of them.
rmv_timeGoTone_if_stimOffset_aft_goTone = 0; 
rmv_time1stSide_if_stimOffset_aft_1stSide = 0; 
setNaN_goToneEarlierThanStimOffset = 0;

stimAftGoToneParams = {rmv_timeGoTone_if_stimOffset_aft_goTone, rmv_time1stSide_if_stimOffset_aft_1stSide, setNaN_goToneEarlierThanStimOffset};
% stimAftGoToneParams = []; % {0,0,0};

[timeNoCentLickOnset, timeNoCentLickOffset, timeInitTone, time1stCenterLick, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, time1stCorrectTry, ...
    time1stIncorrectTry, timeReward, timeCommitIncorrResp, time1stCorrectResponse, timeStop, centerLicks, leftLicks, rightLicks] = ...
    setEventTimesRelBcontrolScopeTTL(alldata, trs2rmv, scopeTTLOrigTime, stimAftGoToneParams, outcomes);


% keep a copy of original time events before changing them.
timeCommitCL_CR_Gotone0 = timeCommitCL_CR_Gotone;
timeStimOnset0 = timeStimOnset;
time1stSideTry0 = time1stSideTry;
timeCommitIncorrResp0 = timeCommitIncorrResp;

set_change_of_mind_trs % set change-of-mind trials. output will be trs_com.


%% Set ep (SVM will be trained on ep frames to decode choice) and set to nan timeStimOnset of those trials 
% that have issues with this ep: ie their go tone is within ep, or their stimDur is not long enough (800ms).

% include in timeStimOnset trials that are:
% stimulus duration >= th_stim_dur 
% and go tone after ep_ms(2)


ep_ms = [500 700]; % rel2 stimOnset % we want to decode animal's upcoming choice by traninig SVM for neural average responses during [500 700]ms after stimulus onset. Why?
% bc we think at this window choice signals might already exist, also we
% can see if they continue during [700 stimOffset], ie mostly [700 1000]ms.
% (ie if decoder generalizes to time points beyond ep)
% you can also try [600 800].... but for now lets go with [500 700].

% now make sure in no trial go tone happened before the end of ep:
i = timeCommitCL_CR_Gotone <= ep_ms(end);
if sum(i)>0
    fprintf('Excluding %i trials from timeStimOnset bc their goTone is earlier than ep end\n', sum(i))
    timeStimOnset(i) = NaN;  % by setting to nan, the aligned-traces of these trials will be computed as nan.
else
    fprintf('No trials with go tone before the end of ep. Good :)\n')
end

% now make sure trials that you use for SVM (decoding upcoming choice from
% neural responses during stimulus) have a certain stimulus duration. Of
% course stimulus needs to at least continue until the end of ep. 
% go with either 900 or 800ms. Since the preference is to have at least
% ~100ms after ep which contains stimulus and without any go tones, go with 800ms
% bc in many sessions go tone happened early... so you will loose lots of
% trials if you go with 900ms.

th_stim_dur = 800; % min stim duration to include a trial in timeStimOnset

figure; hold on
plot(timeCommitCL_CR_Gotone - timeStimOnset)
plot(timeStimOffset - timeStimOnset)
plot([1 length(b)],[th_stim_dur th_stim_dur],'g')
ylabel('Time relative to stim onset (ms)')
legend('stimOffset','goTone', 'th\_stim\_dur')
minStimDurNoGoTone = min(timeCommitCL_CR_Gotone - timeStimOnset); % this is the duration after stim onset during which no go tone occurred for any of the trials.
cprintf('blue', 'minStimDurNoGoTone = %.2f ms\n', minStimDurNoGoTone)


% exclude trials whose stim duration was < th_stim_dur
j = (timeStimOffset - timeStimOnset) < th_stim_dur;
if sum(j)>0
    fprintf('Excluding %i trials from timeStimOnset bc their stimDur-w/out-goTone < 800ms\n', sum(j))
    timeStimOnset(j) = NaN;
else
    fprintf('No trials with stimDur-w/out-goTone < 800ms. Good :)\n')
end


fprintf('#isnan(timeStimOnset0)= %i;  #isnan(timeStimOnset)= %i\n', sum(isnan(timeStimOnset0)), sum(isnan(timeStimOnset)))

% how about trials that go tone happened earlier than stimulus offset?
% I don't think they cause problems for SVM training. Except that when you
% look at projections (neural responses projected onto SVM weights) you
% need to have in mind that go tone may exist before stim ends.
% so we don't exclude them but lets just check what is the duration after
% stim onset in which no go tone occurred for any of the trials.
toRmv = (i+j)~=0;
final_minStimDurNoGoTone = min(timeCommitCL_CR_Gotone(~toRmv) - timeStimOnset(~toRmv)); % for trials that you are including in SVM, this is the duration after stim onset during which no go tone occurred for any of these trials.
fprintf('final_minStimDurNoGoTone = %.2f ms\n', final_minStimDurNoGoTone)


%% Ok now align traces on stimulus

% remember traces_al_sm has nan for trs2rmv as well as trs in alignedEvent that are nan.
alignedEvent = 'stimOn'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'

[traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt...
    (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
    timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);

clear stimAl
stimAl.traces = traces_al_sm;
stimAl.time = time_aligned;
stimAl.eventI = eventI;

epStartRel2Event = round(ep_ms(1)/frameLength); % the start point of the epoch relative to alignedEvent for training SVM. (500ms)
epEndRel2Event = floor(ep_ms(2)/frameLength); % the end point of the epoch relative to alignedEvent for training SVM. (700ms)
stimAl.ep = eventI+epStartRel2Event : eventI+epEndRel2Event; % frames on stimAl.traces that will be used for trainning SVM.

stimAl


%% Now take care of trials you want to use for 1stSideTry alignment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We don't want to include in time1stSideTry, change-of-mind trials, ie
% mouse did a single try on one side, did not commit it, and then tried the
% other side. You can put their aligned traces in a separate array, to
% later study them.

% Remember a trial may go to 1stSideTry, and be followed by reward (mouse
% did incorr try, commit it, then correct it); but if u change response on
% allowCorr it will be counted as incorr outcome.

% Details:
% what trials to include when projecting choice-aligned traces onto decoder
% weights? what is it that you want to get when looking at those
% projections? I want to see how choice-aligned trace projections look on
% the axes that maximally separate neural responses during the stimulus. I
% know at time 0, 1stSideTry happened, and before that we have the go tone
% and the stimulus. Since goTone-1stSideTry interval varies for different
% trials (look at the plot below) I don't think you need to make sure there
% is no stimulus after go tone... so I would accept all time1stSideTry,
% without setting any trials to nan (except for those that are already nan
% due to no choice)

% set_change_of_mind_trs % set change-of-mind trials. output will be trs_com.
if sum(trs_com)>0
    cprintf('blue', 'Excluding %i time1stSideTry bc of change-of-mind\n', sum(trs_com))
    time1stSideTry(trs_com) = nan;
else
    cprintf('blue', 'Not excluding any of time1stSideTry, bc no change-of-mind\n')
end

fprintf('#isnan(time1stSideTry0)= %i; #isnan(time1stSideTry)= %i\n', sum(isnan(time1stSideTry0)), sum(isnan(time1stSideTry)))

% look at goTone to 1stSideTry time:
a = time1stSideTry - timeCommitCL_CR_Gotone;
figure; plot(a)
xlabel('Trial')
ylabel('Time btwn goTone & 1stSideTry (ms)')


%% Align traces on 1stSideTry: excluding change-of-mind trials (ie trying without committing), so these are all followed by a commit lick on the same side.

alignedEvent = '1stSideTry'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'

[traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt...
    (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
    timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);

clear firstSideTryAl
firstSideTryAl.traces = traces_al_sm;
firstSideTryAl.time = time_aligned;
firstSideTryAl.eventI = eventI;
firstSideTryAl


%% Align traces on 1stSideTry: analyze COM trials:

if sum(trs_com)>0
    time1stSideTry = time1stSideTry0;
    time1stSideTry(~trs_com) = NaN;
    alignedEvent = '1stSideTry'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'
    [traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt...
        (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
        timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);

    clear firstSideTryAl_COM
    firstSideTryAl_COM.traces = traces_al_sm;
    firstSideTryAl_COM.time = time_aligned;
    firstSideTryAl_COM.eventI = eventI;    
else
    firstSideTryAl_COM = [];
end
firstSideTryAl_COM



%% Now take care of trials you want to use for goTone alignment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% keep in mind go tone happens together with center lick commit and center
% reward.

% save two sets of traces for goTone-aligned: 1) including all trials. 2)
% excluding trials with stim after go tone.

% details: here we want to see how projections of goTone-aligned traces onto
% dimensions that maximally separate neural responses during stimulus
% (to match the upcoming choice) look like? At time 0 we know go tone
% happened for all trials and before that we know there is stimulus. Lets
% review these again... there could be a delay between stim offset and go
% tone (depends on when the mouse commits the center lick)... fine. How
% about having trials in which stim continues after go tone... is that
% fine? well if the question is how go tone changes the space of neural
% responses, then having or not having stimulus after go tone may not make
% a difference... in a sense you can check for it. you can compare
% projections for all trials vs trials w no stim after go tone... I'm
% saying this bc in some sessions u may not have enough trials if u exclude
% all that have stim after go tone.... so how about this: save two sets of
% traces for goTone-aligned: 1) including all trials. 2) excluding trials
% with stim after go tone. Great.

numTrsGoToneBefStimOffset = sum(timeCommitCL_CR_Gotone <= timeStimOffset);

% what is the time interval between stim offset and go tone?
a = timeCommitCL_CR_Gotone - timeStimOffset;
figure; plot(a)
xlabel('Trial')
ylabel('Time btwn stimeOffset & goTone(ms)')
title(['# trs with stim after go tone = ', num2str(numTrsGoToneBefStimOffset)])



%% Align traces on goTone: include all trials (even if stimulus continued after go tone)

alignedEvent = 'goTone'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'
timeCommitCL_CR_Gotone = timeCommitCL_CR_Gotone0;

[traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt...
    (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
    timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);

clear goToneAl
goToneAl.traces = traces_al_sm;
goToneAl.time = time_aligned;
goToneAl.eventI = eventI;
goToneAl


%% Align traces on goTone: include only trials without stimulus after go tone

if sum(timeCommitCL_CR_Gotone0 <= timeStimOffset)>0
    cprintf('blue', 'Setting goToneAl_noStimAft for %i trials\n',sum(timeCommitCL_CR_Gotone0 <= timeStimOffset))

    % set to nan timeGoTone for trials that have stim after go tone
    timeCommitCL_CR_Gotone(timeCommitCL_CR_Gotone0 <= timeStimOffset) = NaN;  

    fprintf('#isnan(timeCommitCL_CR_Gotone0)=%i; #isnan(timeCommitCL_CR_Gotone)=%i\n', sum(isnan(timeCommitCL_CR_Gotone0)), sum(isnan(timeCommitCL_CR_Gotone)))

    alignedEvent = 'goTone'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'

    [traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt...
        (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
        timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);

    clear goToneAl_noStimAft
    goToneAl_noStimAft.traces = traces_al_sm;
    goToneAl_noStimAft.time = time_aligned;
    goToneAl_noStimAft.eventI = eventI;
    
else % no trials with stim continuing after go tone.
    goToneAl_noStimAft = goToneAl;
end
goToneAl_noStimAft



%% Now take care of trials you want to use for reward alignment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bc allowCorrectResp = 'change', timeReward trials are not preceded by
% incorrect response.


% I don't think there are any trials here that you need to remove. 
% bc u changed outcomes for allowCorrection trials, timeReward will only
% reflect trials that mouse's 1st choice was correct, not the ones that
% mouse committed incorr choice and corrected them... so it will not be
% preceded by incorr resp in any of the trials.

% Align traces on reward

fprintf('#isnan(timeReward) = %i\n', sum(isnan(timeReward)))
fprintf('Not excluding any of the timeReward trials.\n')

alignedEvent = 'reward'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'

[traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt...
    (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
    timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);

clear rewardAl
rewardAl.traces = traces_al_sm;
rewardAl.time = time_aligned;
rewardAl.eventI = eventI;
rewardAl



%% Now take care of trials you want to use for incorrect-response alignment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bc u changed outcomes for allowCorrection trials, timeCommitIncorrResp
% will reflect the 1st committed choice, and it may be followed by a reward
% (in case mouse entered allowCorrection and corrected his choice).


% I am not sure if you need to do the following, but just see how many of
% these trials you have.
% exclude from timeCommitIncorrResp, trials that mouse entered
% allowCorrection, and corrected his choice and received reward.

% trials that the mouse entered the allow correction state. (remember :
% this does not give all trials that the mouse 1st committed error. If the mouse was on sideChoose, he will go to punish (and not punish allow correction).)
a = arrayfun(@(x)x.parsedEvents.states.punish_allowcorrection, alldata, 'uniformoutput', 0);
allowCorrectEntered = ~cellfun(@isempty, a);
% fprintf('%.3f = Fraction of trials aninmal entered allowCorrection\n', nanmean(allowCorrectEntered))
a = sum(allowCorrectEntered & [alldata.outcome]==1);
cprintf('blue', '# allowCorrEntered trs followed by reward = %i\n', sum(a))
cprintf('blue', 'you are not excluding them from timeCommitIncorrResp\n')
% timeCommitIncorrResp(a) = NaN;

fprintf('#isnan(timeCommitIncorrResp) = %i\n', sum(isnan(timeCommitIncorrResp)))


% Align traces on commitIncorrResp

alignedEvent = 'commitIncorrResp'; % align the traces on stim onset. % 'initTone', 'stimOn', 'goTone', '1stSideTry', 'reward', 'commitIncorrResp'

[traces_al_sm, time_aligned, eventI] = alignTraces_prePost_filt...
    (traces, traceTimeVec, alignedEvent, frameLength, dofilter, timeInitTone, timeStimOnset, ...
    timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, nPreFrames, nPostFrames);

clear commitIncorrAl
commitIncorrAl.traces = traces_al_sm;
commitIncorrAl.time = time_aligned;
commitIncorrAl.eventI = eventI;
commitIncorrAl




%%

if save_aligned_traces
    save(postName, '-append', 'stimAl', 'firstSideTryAl', 'firstSideTryAl_COM', 'goToneAl', 'goToneAl_noStimAft', 'rewardAl', 'commitIncorrAl')
end




