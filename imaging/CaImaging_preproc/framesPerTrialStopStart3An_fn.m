function [framesPerTrial, trialNumbers, frame1RelToStartOff, badAlignTrStartCode, framesPerTrial_galvo, trialStartMissing] = ...
    framesPerTrialStopStart3An_fn(binFilename, framecountFilename, headerBug, all_data)
% [framesPerTrial, trialNumbers, frame1RelToStartOff, badAlignments] = ...
%   framesPerTrialStopStart3An(binFilename, framecountFilename [, headerBug])
%
% This goal of this function is to take two inputs (the analog signal
% acquired alongside the calcium imaging, and the framecounts.txt file
% generated by the script running on the scope) and parse them out to
% identify the frames corresponding to each trial.
%
% Expects a .bin file containing 4 analog channels:
%  AI0: trial codes (encoded as a modified "interleaved 2 of 5" barcode)
%  AI1: slow galvo signal
%  AI2: "trial start" signal (36 ms TTL at start of trial)
%  AI3: dummy channel, not used (but must be present to avoid bug in MScan)
%
% This process is a little complicated, because neither BControl nor MScan
% is totally reliable.
%
% On the BControl side, sometimes the start TTL and trial code seem to be
% missing, but there is an interruption in the slow galvo signal that can
% be used to identify trials that need to be broken up.
%
% On the MScan side, the analog signal and the scanning have no intrinsic
% time alignment signals because of bad design in MScan. It would seem that
% the slow galvo signal should be enough to reconstruct the timing.
% However, the slow galvo signal by itself is not adequate because MScan
% usually drops any final partial frame, but not always (sometimes it
% finishes that partial frame out, and saves it like any other frame). So
% instead we use a combination of the frame counts recorded in
% framecounts.txt and the slow galvo signal; framecounts.txt appears to be
% totally reliable (except when BControl messes up; see above).
%
% Trial numbers are aligned using the "bar codes" that get sent over AI0.
% Rarely, these get lost or are mangled. In those cases, they are usually
% repairable, or can be inferred. If this endangers time alignment, those
% trials are marked as having problematic alignment in badAlignments.
%
% INPUTS
%   binFilename        -- path to the .bin file containing the analog
%                         signals
%   framecountFilename -- path to the framecounts.txt file
%   headerBug          -- optional. Should be true if recording was made
%                         using MView 3.?.?.?, which writes slightly broken
%                         file headers. Default false.
%
% OUTPUTS
%   framesPerTrial     -- vector of length nTrials. Contains the number of
%                         frames per trial
%   trialNumbers       -- vector of the trial numbers. If values are
%                         negative, this means that the code was damaged
%                         and could not be inferred even with the help of
%                         surrounding trials
%   frame1RelToStartOff -- when the first frame of this trial started,
%                          relative to the offset of the initial TTL on the
%                          startTrial line
%   badAlignments      -- trials that could not be aligned, due to missing
%                         signals or mismatches between signals. Farzaneh
%                         Note: this variable is not used in
%                         mergeActivityIntoAlldata ...


%% Important notes:

% nTrials: based on trialStart (send by trialStart from bcontrol). trialStart and scopeTTL (beginning of scanning) happen at the same time.

% trialNumbers, codeTimes: based on trialCodes (sent by scopeTrial from bcontrol)

% remember their length is not necessarily equal to the length of the valid
% elements in the frameCount files, because frameCount files most of the
% times shows nan for the last trial (I guess this happens when you pause
% recording during the last trial, so even though trialStart and trialCodes
% are sent, there is no corresponding frame number for the last trial in
% the frameCount file.)


%% Parameters

voltageThresh = 1.5;


%% Optional arguments

if ~exist('headerBug', 'var')
    headerBug = 0;
end


%% Read binary file

volt = readMOMAnalog(binFilename, headerBug);

trialCodes = volt(1, :); % this is related to scopeTrial sent from bcontrol.
slowGalvo = volt(2, :);
trialStart = volt(3, :); % this is related to trialStart sent from bcontrol.

trialStart(trialStart < voltageThresh) = 0;
trialStart(trialStart >= voltageThresh) = 1;


%% Get number of trials from trialStart and make sure all trials have a trialStart signal, 
% otherwise you won't know about mscan delay and proper alignment with behavioral data will be off by a max of 31ms.

% Find trial start and end samples

% protocols where trialStart was sent during state trial_start_rot_scope
if isfield(all_data(1).parsedEvents.states, 'trial_start_rot_scope')
    
    %%
    startOnSamples = find(diff(trialStart) == 1) + 1;
    if trialStart(1) == 1
        startOnSamples = [1 startOnSamples];
    end
    
    startOffSamples = find(diff(trialStart) == -1);
    
    nTrials = length(startOnSamples);
    
    frame1RelToStartOff = NaN(1, nTrials);
    for tr = 1:nTrials
        frame1RelToStartOff(tr) = -(startOffSamples(tr) - startOnSamples(tr) + 1);
    end
    
    
    % startRotScope_trialCode1_dur (same as diffs): interval between the end of
    % trialStart (= beginning of start_rotary_scope) and the beginning of
    % trial code.
    startRotScope_trialCode1_dur = NaN(1, length(all_data)-1); % length(codeTimes)
    for tr = 1:length(all_data)-1 % length(codeTimes)
        startRotScope_trialCode1_dur(tr) = round(1000*all_data(tr).parsedEvents.states.trialcode1(1) - 1000*all_data(tr).parsedEvents.states.start_rotary_scope(1));
    end
    
    
else % protocols where trialStart was sent during state start_rotary_scope
    
    %%
    % Farz commented below. Not applicable to my states.
    %{
    startOnSamples = find(diff(trialStart) == 1) + 1;
    if trialStart(1) == 1
      startOnSamples = [1 startOnSamples];
    end
    startOffSamples = find(diff(trialStart) == -1);
    nTrials = length(startOnSamples);
    %}
    
    if ~isfield(all_data, 'startScopeDur')
        error('startScopeDur needs to be provided!')
    elseif length(unique([all_data.startScopeDur])) > 1
        error('trials have different startScopeDurs!')
    end
    
    dur_state = all_data(1).startScopeDur*1000; % 500; % duration of state start_rotary_scope set in bcontrol.
    
    if all_data(1).iti_min*1000 < dur_state
        error('iti_min is shorter than startScopeDur, hence finding trials from trialStart signal is not easily doable!')
    end
    
    
    on_state = find(diff(trialStart)==1); % the sample (ms, signal low) that the animal entered the state start_rotary_scope. The actual onset of the state start_rotary_scope is 1 sample next.
    off_state = find(diff(trialStart)==-1); % the sample (ms, signal high) that the animal exit the state start_rotary_scope.
    dur_noTTL = on_state - off_state(1:end-1); % dur_noTTL shows the interval that trialStart signal is low, ie the animal was not in state start_rotary_scope. % the last interval (after the last state) is missing here.
    
    % %% debugging
    onoff_state = [[0,on_state]' off_state']; % timepoints (ms) that animal entered (on) and exit (off) the state start_rotary_scope.
    dur_ttl = diff(onoff_state,[],2); % duration (ms) that animal spent in each state start_rotary_scope.
    % [dur_ttl(1:end-1) [dur_noTTL]'] % duration that animal was and was not in start_rotary_scope
    
    
    istr = dur_noTTL > dur_state; % only those states that were followed by a long interval indicate a trial. If the animal bypasses start_rotary_scope and immediately proceeds from start_rotary_scope to the rest of the trial states, dur_noTTL will be for sure > dur_state (bc iti is one of the states of dur >~1sec). On the other hand if the animals went to start_rotary_scope2 after start_rotary_scope, and managed to stay there for dur_state, hence managed to proceeded to the rest of the states, again dur_noTTL will be for sure > dur_state. It will be only problematic if iti_min is < dur_state, in which case you wont be quite sure that an dur_noTTL < dur_state is bc the animal had a very fast trial, or bc the animal had a short start_rotary_scope2, and hence no trial!
    
    startOnSamples = on_state(istr) + 1; % the onset (signal high) of the first start_rotary_scope state right before animal initiated.)
    if trialStart(1) == 1
        startOnSamples = [1 startOnSamples];
    end
    
    startOffSamples = [off_state(istr) off_state(end)]; % the offset (signal high) of the last start_rotary_scope state right before animal initiated. trialCode happens right after here, or dur_state ms after here (if the animal entered the state start_rotary_scope2).
    
    nTrials = sum(istr) + 1; % 1 is added bc interval_statate doesn't include the last interval corresponding to the last trial.
    
    frame1RelToStartOff = -[dur_ttl(1); dur_ttl(find(istr)+1,1)]'; % matt's code equals this +1. % frame1RelToStartOff shows the lag between when bcontrol sent scopeTTL (+ trialStart) and when MScan started scanning. If it is -499 (ie the duration of start_rotary_scope) there was no lag. If it is >-499, it means MScan did not save the entire duration of start_rotary_scope, hence imaging started at a delay. If it <-499, mouse has entered state start_rotary_scope2.
    % frame1RelToStartOff equals duration (in negative sign) of the 1st start_rotary_scope state computed from MScan (trialStart
    % analog channel). If it is less than diff(all_data(tr).parsedEvents.states.start_rotary_scope(1,:)), it indicates lag in MScan starting the
    % scan relative to bcontrol starting the TTL pulse.
    
    % d1 computed below should not be less than -1, or more than ~30ms ...
    % otherwise something has gone with the recording of trialStart sent by the first start_rotary_scope state (due to it being too short and occuring at the middle of scanning of a frame, hence no aqcuiring would heppen).
    % d1 = NaN(1,nTrials-1); for tr=1:nTrials-1, d1(tr) = diff(all_data(tr).parsedEvents.states.start_rotary_scope(1,:))*1000+frame1RelToStartOff(tr); end
    
end

fprintf('%i trialCode signals were found.\n', nTrials)


%% Get trial numbers from trialCodes

[trialNumbers, codeTimes] = segmentVoltageAndReadBarcodes(trialCodes);

fprintf('%i trialCode signals were found.\n', length(trialNumbers))

if length(unique(diff(trialNumbers))) > 1 % ||  unique(diff(trialNumbers))~=1
    error('TrialCodes are not consecutive! potential missing trialCode signals!! \n')
end


%% Check if number of trialStart signals matches length of trialNumbers (coming from trialCode signal).
% If not fix vars related to trialStart

% Important notes:
% nTrials: based on trialStart (send by trialStart from bcontrol). trialStart and scopeTTL (beginning of scanning) happen at the same time.

% trialNumbers, codeTimes: based on trialCodes (sent by scopeTrial from bcontrol)

% remember their length is not necessarily equal to the length of the valid
% elements in the frameCount files, because frameCount files most of the
% times shows nan for the last trial (I guess this happens when you pause
% recording during the last trial, so even though trialStart and trialCodes
% are sent, there is no corresponding frame number for the last trial in
% the frameCount file.)


% Find trials that might miss trialStart signal. Also gives you clue on how to fix the mismatch in size of nTrials and trialNumbers.
trialStartMissingPlot
missedTr = []; % missedTr is the tr for which no trialStart was recorded. trialStartMissing are the trs which had more than one start_rotary_scope but one did not send trialStart signal to mscan.
trialStartMissing = false(size(trialNumbers));


if length(trialNumbers) > nTrials
    warning('framesPerTrialStopStart3An:missingStarts', 'Missing %d trialStart signal(s)', ...
        length(trialNumbers) - nTrials);
    disp('Use trialStartMissingPlot and then trialStartMissingFind to set trialStartMissing trials.')
    
    trialStartMissingFind % use this to set missedTr and trialStartMissing.
    
    trialStartMissing(missedInds) = true;
    
elseif length(trialNumbers) < nTrials
    error('framesPerTrialStopStart3An:missingCodes', 'Missing %d entire trial codes', ...
        length(trialNumbers) - nTrials);
end


% Fix length of vars related to trialStart

if length(missedTr)>1
    error('more that one missedTr (in trialStart signal) found!!!')
end

if ~isempty(missedTr)
    startOnSamples = [startOnSamples(1:missedTr-1)  NaN  startOnSamples(missedTr:end)];
    startOffSamples = [startOffSamples(1:missedTr-1)  NaN  startOffSamples(missedTr:end)];
    frame1RelToStartOff = [frame1RelToStartOff(1:missedTr-1)  NaN  frame1RelToStartOff(missedTr:end)];
    nTrials = length(frame1RelToStartOff);
else
    disp('No trials with missed trialStart exist :-) ')
end



%% Cross-check start-TTL-off times and code-on times

trialStartOff_trialCodeOn_dur = (startOffSamples + 1) - codeTimes; % used to be named diffs. % this is the gap between trialStart offset and trialCode onset (when they were both low). If animal waited in start_rotary_scope for dur_state, we expect this gap to be 0 (since trialCode is sent in the state immediately following start_rotary_scope). If animal went into start_rotary_scope2 and then trialCode state, then this gap should be dur_state (which is the duration of start_rotray_scope2).

% The codes below I think are better and more general for setting badAlignTrStartCode
%{
startRotScope_trialCode1_dur = NaN(1, length(codeTimes));
for tr = 1:length(codeTimes)
    startRotScope_trialCode1_dur(tr) = round(1000*all_data(tr).parsedEvents.states.trialcode1(1) - 1000*all_data(tr).parsedEvents.states.start_rotary_scope(1));
end
badAlignTrStartCode = (isnan(diffs) | abs(startRotScope_trialCode1_dur + diffs) > 1);
%}


if isfield(all_data(1).parsedEvents.states, 'trial_start_rot_scope')
    minl = min(length(startRotScope_trialCode1_dur), length(trialNumbers));
    badAlignTrStartCode = (isnan(trialStartOff_trialCodeOn_dur(1:minl)) | abs(startRotScope_trialCode1_dur(1:minl) + trialStartOff_trialCodeOn_dur(1:minl)) > 1);
    % badAlignments = (abs(diffs) > 1); % Farz commented. not applicable to my states.    
else
    badAlignTrStartCode = ~ismember(trialStartOff_trialCodeOn_dur,[0, -dur_state, -1, -dur_state-1]); % -1 needs check: matt said he allows 1ms for slightly unequal rise and fall times of the different ttl lines.
end

badAlignTrStartCode(missedTr) = false; % since missedTr is the tr without a trialStart, it doesn't make sense to call it a badAlignTrStartCode trial. Instead, missedTr will be marked in the trialStartMissing array.



%{
stateOff_codeOn_samples = sort([off_state, codeTimes]);
indCodeOn = find(ismember(stateOff_codeOn_samples, codeTimes));
diffs = codeTimes - stateOff_codeOn_samples(indCodeOn-1);
badAlignments = ~ismember(diffs,[1, dur_state+1]);
%}

if any(badAlignTrStartCode)
    warning('In trial %d start-off samples did not line up with code onsets by more than one sample.\n', find(badAlignTrStartCode));
end

% %% Debugging
%
% figure;
% hold on;
%
% plot(slowGalvo);
% plot(startOnSamples, slowGalvo(startOnSamples), 'r.');
% plot(startOnSamples-1, slowGalvo(startOnSamples-1), 'g.');


%% Read the frameCount text file to get number of frames per trial.

if exist('framecountFilename', 'var') && ~isempty(framecountFilename)    
    
    framecountFrames = NaN(1, nTrials);
    
    % Open file
    try
        fid = fopen(framecountFilename, 'r');
    catch
        warning('Could not open frame count file %s', framecountFilename);
        return;
    end
    
    % Read each line, keep only the initial number
    line = 0;
    while 1
        line = line + 1;
        oneLine = fgetl(fid);
        if ~ischar(oneLine) || isempty(oneLine)
            break;
        end
        
        tokens = simpleTokenize(oneLine, ' ');
        numStr = tokens{1};
        framecountFrames(line) = str2double(numStr);
    end
    

else
    warning('No framecounts.txt supplied. Alignment is very likely to drift over trials!');
   
end






%%%%%%%%%%%%%%%% The following codes are un-necessary! %%%%%%%%%%%%%%%%


%% Use the slow galvo signal to identify frames

trialEndSamples = [startOnSamples(2:end)-1 length(slowGalvo)];
framesPerTrial = cell(1, nTrials);
% frame1RelToStartOff = NaN(1, nTrials);
for tr = 1:nTrials
    if ~isnan(startOnSamples(tr)+trialEndSamples(tr))
        framesPerTrial{tr} = parseSlowGalvo(slowGalvo(startOnSamples(tr):trialEndSamples(tr)));
    end
    %   frame1RelToStartOff(tr) = -(startOffSamples(tr) - startOnSamples(tr)); % Farz commented. not applicable to my states.
end


% %% Debugging
%{
% a = [framecountFrames' framesPerTrial'];
% framedur = .032363833;
% [a, diff(a,[],2) framedur*diff(a,[],2)*1000]
%}

%% If we found any "trial" was actually more than one trial, deal with it

% Sometimes, we see a brief pause in the galvo trace, then it resumes
% without seeing a code or trial start. This might be due to when BControl
% is paused and unpaused. In any case, we need to fix these, because
% otherwise we'll match up trial codes with trials wrong.

multChunks = find(cellfun(@length, framesPerTrial) > 1);
if ~isempty(multChunks)
    for m = fliplr(multChunks)
        if m == 1
            frame1RelToStartOff = [frame1RelToStartOff(1:m) NaN(1, length(framesPerTrial{m})-1) ...
                frame1RelToStartOff(m+1:end)];
            trialNumbers = [trialNumbers(1:m) NaN(1, length(framesPerTrial{m})-1) ...
                trialNumbers(m+1:end)];
            framesPerTrial = [num2cell(framesPerTrial{m}) framesPerTrial(m+1:end)];
        elseif m == length(framesPerTrial)
            frame1RelToStartOff = [frame1RelToStartOff(1:m) NaN(1, length(framesPerTrial{m})-1)];
            trialNumbers = [trialNumbers(1:m) NaN(1, length(framesPerTrial{m})-1)];
            framesPerTrial = [framesPerTrial(1:m-1) num2cell(framesPerTrial{m})];
        else % Farzaneh: this doesn't give correct answers.
            %{
      frame1RelToStartOff = [frame1RelToStartOff(1:m) NaN(1, length(framesPerTrial{m})-1) ...
        frame1RelToStartOff(m+1:end)];
      trialNumbers = [trialNumbers(1:m) NaN(1, length(framesPerTrial{m})-1) ...
        trialNumbers(m+1:end)];
      framesPerTrial = [framesPerTrial(1:m-1) num2cell(framesPerTrial{m}) framesPerTrial(m+1:end)];
            %}
        end
    end
    
    fprintf('%d trial(s) found tacked on to ends of known trials and repaired, but missing alignment info\n', ...
        sum(isnan(frame1RelToStartOff)));
end

framesPerTrial = [framesPerTrial{:}];
framesPerTrial_galvo = framesPerTrial; % Farzaneh


%% Compare galvo #frames with framecounts driven from mscan text file.
    
    if length(framecountFrames) ~= length(framesPerTrial)
        %     error('framesPerTrialStopStart3An:trialCountMismatch', ...
        %       ['Found %d trials in framecounts.txt, but %d trials in the galvo trace. ' ...
        %       'Cannot determine which trial has which frame count!'], ...
        %       length(framecountFrames), length(framesPerTrial)); % Farzaneh commented bc using the galvo trace for sync is just a waste of time...
        
        warning('framesPerTrialStopStart3An:trialCountMismatch', ...
            'Found %d trials in framecounts.txt, but %d trials in the galvo trace. ', ...
            length(framecountFrames), length(framesPerTrial)); % Farzaneh
    end
    
    %   % Compare with frame counts from the analog channels
    %   figure;
    %   hold on;
    %   plot(framecountFrames, 'b');
    %   plot(framesPerTrial, 'r');
    %
    %   figure;
    %   plot(framecountFrames - framesPerTrial, 'k');
    
    % Summarize
    fprintf('For reference (not a problem): galvo trace missing %d frames.\n', nansum(framecountFrames) - nansum(framesPerTrial));
    



% %% Plot some galvo traces with inferred frames, to see what's up
%
% samplesPerFrame = 1000 / 30.9;  % Is this value exact?
%
% trsWrong = find(framecountFrames - framesPerTrial ~= 0, 3);
%
% trsRight = find(framecountFrames - framesPerTrial == 0, 3);
%
% % This is all uncorrected for chunk splits!
% for tr = trsWrong
%   figure;
%   hold on;
%   plot(slowGalvo(startOnSamples(tr):trialEndSamples(tr)));
%   for t = 1:framesPerTrial(tr)
%     plot((1 + t * samplesPerFrame) * [1 1], [-3 3], 'r-');
%   end
%   title(sprintf('%d inferred (wrong)', framesPerTrial(tr)));
% end
%
% for tr = trsRight
%   figure;
%   hold on;
%   plot(slowGalvo(startOnSamples(tr):trialEndSamples(tr)));
%   for t = 1:framesPerTrial(tr)
%     plot((1 + t * samplesPerFrame) * [1 1], [-3 3], 'r-');
%   end
%   title(sprintf('%d inferred (right)', framesPerTrial(tr)));
% end

% %% List the frames where there was disagreement between the galvo signal and framecounts.txt
% % Empirically, there's nothing obviously wrong with these frames.
%
% cumFrames = cumsum(framecountFrames);
% disagreementTrials = find((framecountFrames - framesPerTrial) ~= 0);
% fprintf('Disagreements between galvo signal and framecounts:\n');
% for d = disagreementTrials
%   fprintf('%d\n', cumFrames(d));
% end



%% Go with framecounts.txt

% framecounts.txt gives the more reliable answer
if exist('framecountFilename', 'var') && ~isempty(framecountFilename)
    framesPerTrial = framecountFrames;
end




function nFrames = parseSlowGalvo(galvo)

% First get rid of any initial non-scanning
if galvo(1) == galvo(2) && galvo(1) == galvo(3)
    firstRealValue = find(galvo ~= galvo(1), 1);
    galvo = galvo(firstRealValue:end);
end

% First, look for whether there are breaks in the middle of scanning,
% indicating trials that weren't separated correctly.

chunks = breakScanningChunks(galvo);

samplesPerFrame =  32.363833; % Farzaneh: This value is shown (as frame duration) in the File Properties button of File Manager window of Mview, apparently more accurate than 1000/30.9Hz. % 1000 / 30.9;  % Is this value exact?

nFrames = cellfun(@(c) floor(length(c) / samplesPerFrame), chunks); % Matt is getting floor, I guess bc if the ttl is lowered at the middle of a frame, the imaging frame will be dropped, but the analog data will be saved (upto that point). However there are times that the imaging data gets saved through the remainder of the frame (although the analog stops right when ttl is lowered). These cases are why sometimes you get 1 less frame in framesPerTrial comapred to framecountFrames.




function chunks = breakScanningChunks(galvo)
% This function looks to see whether scanning stopped in the middle of what
% should be a single trial. If so, break it up.

% This should happen rarely, so it's ok to use a slightly inefficient
% recursive algorithm for convenience.

% galvo = [galvo, 0 0 0]; % Farzaneh added this, bc some trials lack identical values at the end and they "chunk" value will come out empty.
% To find scan stops, we'll look for 3 consecutive identical values not at
% the end.
runs = (galvo(1:end-2) == galvo(2:end-1) & galvo(1:end-2) == galvo(3:end));
firstRun = find(runs, 1);
lastNonRun = find(runs == 0, 1, 'last'); % Farz commented.
% lastNonRun = find(runs(1:end-3) == 0, 1, 'last'); % Farzaneh, bc 0 0 0 is added to the end.
if isempty(firstRun) || isempty(lastNonRun) || firstRun == lastNonRun + 1
    chunks = {galvo(1:firstRun - 1)};
else
    endOfFirstRun = firstRun + find(runs(firstRun+1:end) == 0, 1);
    chunks = [{galvo(1:firstRun - 1)} breakScanningChunks(galvo(endOfFirstRun:end))];
end



%% Farzaneh tests : compare the duration of trirals in bcontrol and mscan
%{
% load alldata
frameLength = 1000 / 30.9;

clear nfrs dur_nottl
for tr = 2:length(all_data)-1
    % duration of no scopeTTL, preceding the trial
    dur_nottl(tr) = all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000 - all_data(tr-1).parsedEvents.states.stop_rotary_scope(1)*1000;
end
dur_nottl(1) = NaN;
[m, i] = min(dur_nottl)

for tr = 1:length(all_data)-1
    % duration of a trial in mscan (ie duration of scopeTTL being sent).
    durtr = all_data(tr).parsedEvents.states.stop_rotary_scope(1)*1000 + 500 - ...
        all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000; % 500 is added bc the duration of stop_rotary_scope is 500ms.
    nfrs(tr) = durtr/ frameLength;
end
size(nfrs)

% duration of state check_next_trial_ready
% all_data(tr+1).parsedEvents.states.state_0(2)*1000 - all_data(tr).parsedEvents.states.stop_rotary_scope(1)*1000


%%
% trials that miss a trial code (ie were not recorded in mscan but recorded
% in bcontrol)
trmiss = find(~ismember(1:length(trialNumbers), trialNumbers))
nfrs(trmiss) = [];

%%
size(cell2mat(framesPerTrial)')
f = cell2mat(framesPerTrial)';

[(1:length(framecountFrames))' nfrs' framecountFrames' f(1:length(framecountFrames))]

%}

