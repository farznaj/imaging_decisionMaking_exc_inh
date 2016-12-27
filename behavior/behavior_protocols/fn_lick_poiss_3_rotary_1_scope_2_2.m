% Version comparison: 2_2 vs 2 : in 2_2 you added the state
% trial_start_rot_scope which lasts for 36ms and sends the trialStart
% signal. In version 2, trial start was sent during start_rotary_scope
% which had variable duration, hence causing complexity in data analyis.
% (written on 151209).


%  fn_lick_1_poiss_2 revisions (Mar 2015)
    % waitDur and sideLickDur will be changed adaptively if mouse doesn't
    % do well on a number of consecutive trials.
    
    % create_visual_signal_fn (and audio) modified so that no
    % extra event is anymore added to the end of the stimulus. This prevents changing the
    % stimulus rate due to this additional event. Also, now the number of events equals the number of intervals. 
    
% fn_lick_1_poiss_2 revisions (Feb 2015)
    % iti state is updated. Now it happens at the beginning of a trial.
    % Therefore value(ITI) indicates iti from the previous trial. The
    % reason for doing this is that if state iti happens at
    % the end of the state matrix, it will be followd by
    % check_next_trial_ready state, over which we cannot have control. As a
    % result even though I was preventing the mouse from licking during
    % iti, it was not effective right before trial initiation (the main time!!)
    % since that was when check_next_trial_ready was happening. 

    % 2/26/15 fn06-fn07 and later on
    % wrong_initiation : start tone comes and mouse side licks. if mouse
    % doesn't lick any of the spouts during wrong_initiation, then this
    % state will keep repeating.
    
        % Remember ITI if computed from the time
        % of reward to the start of the next trial will be stimDur_aftRew sec longer that value(ITI).
        % also remember that ITI does not include the duration of
        % check_next_trial_ready, which is about 1.5 sec for my protocol.
        
        
% fn_lick_1_poiss_1 revisions (Feb 2015)
    % 1. Poisson stimulus is used. So, all the parts related to stimulus generation are modified
    
    % 2. Reward stage could be anything and playStimulus could be anything, independent of each other; just set them in the GUI. (before, allow correction was always linked to not playing the stim.)
    
    % 3. stimulus keeps playing for an additional 1sec after the reward is
    % delivered. If in allow correction, stimulus will contintue after the
    % wrong choice. If in only correct (choose side), stimulus will stop
    % immediately after the wrong choice. This was added to help mouse with
    % making stim-reward association.
    
    % 4. a choice (side lick) is defined as licking for .2sec, i.e. when
    % they lick on the side it wont be considered correct or incorrect
    % unless they lick again on that same side after .2sec. This was added
    % to make choices more costly and avoid impulsive choices.
    
    % 5. you can choose what percentage of trials you want to be on allow
    % correction and what percentage on choose side.
    
    
% fn_lick_1_3 revisions (Feb 2015):
    % 1. reward them on the center spout if mouse waited for waitdur.
    % 2. You can avoid having more than th_consectuive trials in a row of
    % the same type. To do so, set th_consecutive to a value > 0 in this
    % script before you start the protocol.
    
    
% Jan 2015: 
        % Farzaneh add the following. The idea is that if waitDur is too short, there is no way for 
        % the animal to make the association, given our type of stimulus. 
        % So at short wait durations, no stimulus will be presented. 
        % The goal for the animal at these short waitDurs is to learn 
        % what initiation tone and go tone mean, and to wait long
        % enough.
    
        
% Dec 2014: Farzaneh modified the state matrix section for the lick
% protocol


% Farzaneh: remember you changed the time units in AllData to sec,  so it matches parsedEvents. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Original authors: David Raposo, John Sheppard
% Copied from dr_wait_training on 05-22-2013.
% Incorporated code from cue_reliability_v5 on 05-27-2013.
% Field names changed to match neural data structures
% Added option for an extra long stimulus (for mice only, in principle)
% Added option to enable/disable wait-start & go tones

% Kachi from June 2013-Present
% Kachi Training Protocol

% June 2013 - Added option to enable LED at water ports

% July 2013 - Added wait period feature in Direct Reward such that trial only initiated
% until reward collected

% July 2013 - Added punishment tone for incorrect choices, in addition to time out -

% Aug 2013(1) - Extended stimulus duration feature. The stimulus can be repeated for as
% long as the experimenter/subject desires. A 1second stimulus is first
% generated and repeated for as long as specified by ExtraStimDuration
% Also added ExtraStimDurationStep to decrease extra stimulus duration over
% time
% Aug 2013(2) - Stimulus Strength proportions feature (beta). Now the user can
% specify the proportion of trials for any desired stimulus strength.
% Commented out the "Modality"-Rates section and replaced with
% "Modality"-RatesProp, which is a 4 element vector that represents the
% proportion for each stimulus strength: easy (9 and 16 evts/s), easy-medium (10 and 15 evts/s) hard-medium (11 and 14 evts/s),
% and hard (12 and 13 evts/s). Ex. for an all easy condition enter: 1 0 0 0.
% Or a mix of easy and hard trials enter: 0.82 0.1 0.08 0 for the
% respective modality
% The sum of the elements in the vector must equal one for adequate
% proportion representation. I would like to add a conditon such that
% harder trials are introduced later in the session. i.e. say the first 50
% or so rewards were based on easy trials only, and subsequent trials after
% this would have a few diffult trials. ..will have to think about how to
% implement this. However it may not be necessary as this feature can be
% updated online, while the animal is doing the task.

% Sept 2013 - removed punishment tone for direct reward, animals will be
% given only time out

% Dec 2013 (1) - updated stim_strength_list function to generate list of possible
% stimulus strengths  using multinomial distribution.

% Jan 2014-antibias from Matt Kaufman. also added start tone flag, as well
% as port light punish flag



function [obj] = fn_lick_poiss_3_rotary_1(varargin)

obj = class(struct, mfilename, pokesplot,soundmanager,soundui,water,saveload,sidesplot,sessionmodel);

%---------------------------------------------------------------
%   BEGIN SECTION COMMON TO ALL PROTOCOLS, DO NOT MODIFY
%---------------------------------------------------------------

% If creating an empty object, return without further ado:
if nargin==0 || (nargin==1 && ischar(varargin{1}) && strcmp(varargin{1}, 'empty')),
    return;
end;

if isa(varargin{1}, mfilename), % If first arg is an object of this class itself, we are
    % Most likely responding to a callback from
    % a SoloParamHandle defined in this mfile.
    if length(varargin) < 2 || ~isstr(varargin{2}),
        error(['If called with a "%s" object as first arg, a second arg, a ' ...
            'string specifying the action, is required\n']);
    else action = varargin{2}; varargin = varargin(3:end);
    end;
else % Ok, regular call with first param being the action string.
    action = varargin{1}; varargin = varargin(2:end);
end;

if ~ischar(action), error('The action parameter must be a string'); end;

GetSoloFunctionArgs(obj);

%---------------------------------------------------------------
%   END OF SECTION COMMON TO ALL PROTOCOLS, MODIFY AFTER THIS LINE
%---------------------------------------------------------------

% ---- From here on is where you can put the code you like.
%
% Your protocol will be called, at the appropriate times, with the
% following possible actions:
%
%   'init'     To initialize -- make figure windows, variables, etc.
%
%   'update'   Called periodically within a trial
%
%   'prepare_next_trial'  Called when a trial has ended and your protocol is expected
%              to produce the StateMachine diagram for the next trial;
%              i.e., somewhere in your protocol's response to this call, it
%              should call "dispatcher('send_assembler', sma,
%              prepare_next_trial_set);" where sma is the
%              StateMachineAssembler object that you have prepared and
%              prepare_next_trial_set is either a single string or a cell
%              with elements that are all strings. These strings should
%              correspond to names of states in sma.
%                 Note that after the 'prepare_next_trial' call, further
%              events may still occur in the RTLSM while your protocol is thinking,
%              before the new StateMachine diagram gets sent. These events
%              will be available to you when 'trial_completed' is called on your
%              protocol (see below).
%
%   'trial_completed'   Called when 'state_0' is reached in the RTLSM,
%              marking final completion of a trial (and the start of
%              the next).
%
%   'close'    Called when the protocol is to be closed.
%
%
% VARIABLES THAT DISPATCHER WILL ALWAYS INSTANTIATE FOR YOU IN YOUR
% PROTOCOL:
%
% (These variables will be instantiated as regular Matlab variables,
% not SoloParamHandles. For any method in your protocol (i.e., an m-file
% within the @your_protocol directory) that takes "obj" as its first argument,
% calling "GetSoloFunctionArgs(obj)" will instantiate all the variables below.)
%
%
% n_done_trials     How many trials have been finished; when a trial reaches
%                   one of the prepare_next_trial states for the first
%                   time, this variable is incremented by 1.
%
% n_started trials  How many trials have been started. This variable gets
%                   incremented by 1 every time the state machine goes
%                   through state 0.
%
% parsed_events     The result of running disassemble.m, with the
%                   parsed_structure flag set to 1, on all events from the
%                   start of the current trial to now.
%
% latest_events     The result of running disassemble.m, with the
%                   parsed_structure flag set to 1, on all new events from
%                   the last time 'update' was called to now.
%
% raw_events        All the events obtained in the current trial, not parsed
%                   or disassembled, but raw as gotten from the State
%                   Machine object.
%
% current_assembler The StateMachineAssembler object that was used to
%                   generate the State Machine diagram in effect in the
%                   current trial.
%
% Trial-by-trial history of parsed_events, raw_events, and
% current_assembler, are automatically stored for you in your protocol by
% dispatcher.m. See the wiki documentation for information on how to access
% those histories from within your protocol and for information.

% edit settings here

% time the valves should be open to deliver 4 uL
left_water_valve_duration = 0.02; 
right_water_valve_duration = 0.02; 
center_water_valve_duration = 0.003;

% Durations for each module of the trial code sent over the TTL line to the microscope
codeModuleDurs = [0.0015 0.0035];



cled = Settings('get', 'DIOLINES', 'center1led');
rled = Settings('get', 'DIOLINES', 'right1led');
lled = Settings('get', 'DIOLINES', 'left1led');

% otherled = Settings('get', 'DIOLINES', 'vision2pulse');

cwater = Settings('get', 'DIOLINES', 'center1water');
rwater = Settings('get', 'DIOLINES', 'right1water');
lwater = Settings('get', 'DIOLINES', 'left1water');

ardttl1 = Settings('get', 'DIOLINES', 'mightexled1'); % DIO1 on channel 4. send a TTL pulse from this channel to pin 51 on arduino.

scopeTTL = Settings('get', 'DIOLINES', 'scopettl');     % This is for turning on scanning
scopeTrial = Settings('get', 'DIOLINES', 'scopetrial'); % This is for sending the trial code

trialStart = Settings('get', 'DIOLINES', 'trialstart'); % For alignment of trial start with scope


% scopetrial
% rotaryCenterVal = 1024;
rotaryResolution = 2048; % X2 encoding, ie reading two counts per segment on the rotary, hence 2*1024 is the resolution.
rotarySampleInt = 10;  % interval between samples of the rotary encoder, in ms
postTrialToRecord = 0; % 1000; % ms after the last ardttl1 pulse during which the rotary's movement will be still recorded.

switch action,
    
    %---------------------------------------------------------------
    %          CASE INIT
    %---------------------------------------------------------------
    
    case 'init'
        
        % Load speaker calibration data.
        load('C:/ratter/ExperPort/WhiteNoiseCalibrationCurrent.mat');
        % Fit a linear model for speaker attenuation and target sound levels (dB SPL).
        SoloParamHandle(obj, 'WhiteNoiseLinearModelParams', 'value', polyfit(reshape(TargetSPLs,1,[]),reshape(10*log10(NoiseAmplitudes),1,[]),1));
        
        load('C:/ratter/ExperPort/ToneCalibrationCurrent.mat');
        % Fit a linear model for speaker attenuation and target sound levels (dB SPL).
        SoloParamHandle(obj, 'ToneLinearModelParams', 'value', polyfit(reshape(TargetSPLs,1,[]),reshape(10*log10(ToneAmplitude),1,[]),1));
        SoloParamHandle(obj, 'AllData', 'value', []);
        
        % gui starts here
        SoloParamHandle(obj, 'mygui', 'saveable', 0);

        mygui.value = figure;
        set(value(mygui), 'Name', mfilename, 'Tag', mfilename, ...
            'closerequestfcn', 'dispatcher(''close_protocol'')', 'MenuBar', 'none');
        
        %-->modified Aug 2013
        screensize = get(0,'screensize');
        set(value(mygui), 'Position', [screensize(3)/4 50 0.75*screensize(3) 0.9*screensize(4)]);
        %         set(value(mygui), 'Position', [500 300 900 830]);
        
        xpos = 10; ypos = 30;        
        EditParam(obj, 'SubjectName', 'subject_name', xpos, ypos); next_row(ypos);
        EditParam(obj, 'Species', 'mouse', xpos, ypos); next_row(ypos);
        EditParam(obj, 'RightWater', right_water_valve_duration, xpos, ypos); next_row(ypos);
        EditParam(obj, 'LeftWater', left_water_valve_duration, xpos, ypos); next_row(ypos);
        EditParam(obj, 'CenterWater', center_water_valve_duration, xpos, ypos); next_row(ypos); % Farz added.
        EditParam(obj, 'RewardVolumeSide', 3, xpos, ypos); next_row(ypos);
        EditParam(obj, 'RewardVolumeCenter', 0.5, xpos, ypos); next_row(ypos);
        
        PushbuttonParam(obj, 'loadsets', xpos, ypos, 'label', 'Load Settings (most recent)');next_row(ypos);
        set_callback(loadsets, {mfilename, 'load_settings_recent'});
        PushbuttonParam(obj, 'loadsets', xpos, ypos, 'label', 'Load Settings (choose file)');next_row(ypos);
        set_callback(loadsets, {mfilename, 'load_settings_file'});        
        PushbuttonParam(obj, 'savesets', xpos, ypos, 'label', 'Save Settings');
        set_callback(savesets, {mfilename, 'save_settings'}); next_row(ypos);
        
        ypos = ypos + 5;        
        DispParam(obj, 'lo_hi_1', '', xpos, ypos); next_row(ypos);
        DispParam(obj, 'lo_hi_2', '', xpos, ypos); next_row(ypos);
        DispParam(obj, 'lo_hi_3', '', xpos, ypos); next_row(ypos);
        DispParam(obj, 'lo_hi_4', '', xpos, ypos); next_row(ypos);        
        EditParam(obj, 'lo_hi_input', '', xpos, ypos); next_row(ypos);        
        DispParam(obj, 'CorrectSideName', '', xpos, ypos); next_row(ypos);
        DispParam(obj, 'eventNumber_VA', '', xpos, ypos); next_row(ypos);
        DispParam(obj, 'mn_cb_mx', '', xpos, ypos); next_row(ypos);   
        EditParam(obj, 'HighRateChoicePort','R',xpos,ypos); next_row(ypos);
        DispParam(obj, 'WaterReceived', 0, xpos, ypos); next_row(ypos);
        DispParam(obj, 'TrialNum', 0, xpos, ypos); next_row(ypos);
        
%         next_row(ypos);        
        xpos = xpos + 220; ypos = 30;       
        DispParam(obj, 'RecordingSession', 0, xpos, ypos); next_row(ypos);
        SoloParamHandle(obj, 'CheckedRecordingSessionValue', 'value', 0);
        
%         next_row(ypos);
%         EditParam(obj, 'LightPunish', 0, xpos, ypos); next_row(ypos); % 1 %added jan 27 2014, uses port LEDs as punishment
%         EditParam(obj, 'PortLED', 0, xpos, ypos); next_row(ypos); % NOT USED..lights up led for correct port, user specifies how long the port led should be on
        EditParam(obj, 'CategoryBoundaryHz', 16, xpos, ypos); next_row(ypos); % 12.5 % category boundary in Hz, i.e. number of events for 1000ms stim.
        EditParam(obj, 'PlayStimulus', 1, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PlayGoTone', 1, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PlayStartTone', 1, xpos, ypos); next_row(ypos); % 1
        SoloParamHandle(obj, 'playStimOrig', 'value', 1);
        
        EditParam(obj, 'AudStrengthPropHME', '0 0 0 0', xpos, ypos); next_row(ypos); % AudioRatesProp
        EditParam(obj, 'VisStrengthPropHME', '0 0 0 1', xpos, ypos); next_row(ypos); % VisualRatesProp
        %         EditParam(obj, 'MultisensoryRatesProp', '0 0 0 0', xpos, ypos); next_row(ypos);        
%         next_row(ypos);        
        EditParam(obj, 'PropOnlyAudio', 0, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PropOnlyVisual', 1, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PropSynchMultisensory', 0, xpos, ypos); next_row(ypos);
        
%         next_row(ypos);        
        DispParam(obj, 'TotalDidNotInitiate', 0, xpos, ypos); next_row(ypos);
        DispParam(obj, 'TotalEarlyDecisions', 0, xpos, ypos); next_row(ypos);
        DispParam(obj, 'TotalDidNotLickAgain', 0, xpos, ypos); next_row(ypos);
        DispParam(obj, 'TotalDidNotChoose', 0, xpos, ypos); next_row(ypos);
        DispParam(obj, 'TotalIncorrect', 0, xpos, ypos); next_row(ypos);
        DispParam(obj, 'TotalCorrect', 0, xpos, ypos); next_row(ypos);  
        EditParam(obj, 'Exclude', 0, xpos, ypos); next_row(ypos); % Farzaneh. if 1 then exclude the current trial from analysis.
        EditParam(obj, 'CenterPoke_amount', 0, xpos, ypos); next_row(ypos); % 5 % set to 0 if you don't want to poke mice with a large drop in the center after some while of not working. set to >0 to determine the amount of water that will be delivered (times the reward amount). give_center_unconst: % give a large drop of water in the center if the mouse did not initiate a trial in [180 300] seconds. water amount = water_duration_ave*value(CenterPoke_amount)
        EditParam(obj, 'CenterPoke_when', [50 180], xpos, ypos); next_row(ypos); % [180 300] a random number will be picked from this interval. if mouse doesn't do anything during wait_for_initiation state for this long, then he will be poked by a large drop on the center.
        SoloParamHandle(obj, 'triExcludeStarted', 'value', []);
        SoloParamHandle(obj, 'adaptiveDurs_force', 'value', 0);
        
        xpos = xpos + 220; ypos = 30;        
        EditParam(obj, 'PostDelayLambda', 45, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PostDelayMax', 0.001, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PostDelayMin', 0, xpos, ypos); next_row(ypos);
        DispParam(obj, 'PostStimDelay', '', xpos, ypos); next_row(ypos);
        
%         next_row(ypos);        
        EditParam(obj, 'PreDelayLambda', 45, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PreDelayMax', 0.001, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PreDelayMin', 0, xpos, ypos); next_row(ypos);
        DispParam(obj, 'PreStimDelay', '', xpos, ypos); next_row(ypos);
        
%         next_row(ypos);
        EditParam(obj, 'StimDur_aftRew', 1, xpos, ypos); next_row(ypos);
        EditParam(obj, 'StimDur_aftRewStep', 0.0001, xpos, ypos); next_row(ypos);
        EditParam(obj, 'ExtraStimDuration', 20, xpos, ypos); next_row(ypos); %.5
        EditParam(obj, 'ExtraStimStep', 0.0001, xpos, ypos); next_row(ypos);        
        EditParam(obj, 'WaitDuration', 0.0250, xpos, ypos); next_row(ypos);
        EditParam(obj, 'WaitDurationStep', 0.0001, xpos, ypos); next_row(ypos); % 0.0003        
        EditParam(obj, 'SideLickDur', 0.12, xpos, ypos); next_row(ypos);
        EditParam(obj, 'SideLickDurStep', 0.0003, xpos, ypos); next_row(ypos);
        EditParam(obj, 'StimDur_diff', 0, xpos, ypos); next_row(ypos); % Farzaneh: stimDur and waitDur are different? (if non zero, then its value will be used as the stimulus duration; however waitDuration will be used to determin with the go cue comes. If 0, then stimulus duraiton will be same as waitDuration (the traditional case).
        SoloParamHandle(obj, 'ExtraStimDurOrig', 'value', []);
        EditParam(obj, 'CenterAgain', .3, xpos, ypos); next_row(ypos);
        EditParam(obj, 'SideAgain', .3, xpos, ypos); next_row(ypos);
        
%         next_row(ypos);        
        xpos = xpos + 220; ypos = 30;
        EditParam(obj, 'MaxInterval', 0.250, xpos, ypos); next_row(ypos);
        EditParam(obj, 'MinInterval', 0.032, xpos, ypos); next_row(ypos);
        EditParam(obj, 'EventDuration', 0.005, xpos, ypos); next_row(ypos); % .015
        EditParam(obj, 'Brightness', 30, xpos, ypos); next_row(ypos);
        MenuParam(obj, 'SoundCarrier', {'White noise' '15 KHz + envelope' '15 KHz'}, 1, xpos, ypos); next_row(ypos);
        EditParam(obj, 'SoundLoudness', 70, xpos, ypos); next_row(ypos);
        EditParam(obj, 'NoiseMaskAmp', 0.0001, xpos, ypos); next_row(ypos);
        MenuParam(obj, 'PunishNoise', {'White noise' 'Pure tone'}, 1, xpos, ypos); next_row(ypos);                
        EditParam(obj, 'PNoiseLoudness', 70, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PunishNoiseDuration', 0, xpos, ypos); next_row(ypos); % it was 2, FN changed to .5 sec.
        EditParam(obj, 'StartGoToneLoudness', 70, xpos, ypos); next_row(ypos);
        MenuParam(obj, 'RewardStage', {'Direct' 'Allow_Choose'}, 2, xpos, ypos); next_row(ypos);
        EditParam(obj, 'PercentAllow', 1, xpos, ypos); next_row(ypos); %5
        DispParam(obj, 'rewardStage', '', xpos, ypos); next_row(ypos);
        EditParam(obj, 'errorTimeout', 2, xpos, ypos); next_row(ypos);
        EditParam(obj, 'TimeToChoose2', 4, xpos, ypos); next_row(ypos); %time allowed to make a decision after the animal side licks but not repeatedly (ie he doesn't sidelick again after sideLickDur)
        EditParam(obj, 'TimeToChoose', 10, xpos, ypos); next_row(ypos); %5        
        EditParam(obj, 'TimeToInitiate', 5, xpos, ypos); next_row(ypos); 
        SoloParamHandle(obj, 'PercentAllowOrig', 'value', []);
        

%         next_row(ypos);  
        xpos = xpos + 220; ypos = 30;               
        EditParam(obj, 'ITI_lambda', 1/3, xpos, ypos); next_row(ypos);
        EditParam(obj, 'ITI_max', 5, xpos, ypos); next_row(ypos);
        EditParam(obj, 'ITI_min', 1, xpos, ypos); next_row(ypos);
        DispParam(obj, 'ITI', '', xpos, ypos); next_row(ypos);

        next_row(ypos); 
        EditParam(obj, 'NumWarmUpTrials', 20, xpos, ypos); next_row(ypos);
        EditParam(obj, 'WaitdurWarmup', 0, xpos, ypos); next_row(ypos);
        EditParam(obj, 'maxInitialBad', 2, xpos, ypos); next_row(ypos); % At the beginning of the session: maximum number of past trials with "bad" outcome, in order to make waitDur short.
        EditParam(obj, 'adaptiveDurs', 0, xpos, ypos); next_row(ypos);
        EditParam(obj, 'maxPastBad', 4, xpos, ypos); next_row(ypos); % maximum number of past trials with "bad" outcome, in order to make waitDur short.
        EditParam(obj, 'PropLeftTrials', 0.5, xpos, ypos); next_row(ypos);     
        SoloParamHandle(obj, 'WaitdurWarmup_noMore', 'value', 0); % we want waitDurWarmup to happen at the beginning of a session but stop once it happened.
        
        % initialize adaptiveDur-related parameters
        SoloParamHandle(obj, 'waitDurOrig', 'value', NaN);
        SoloParamHandle(obj, 'waitDurStepOrig', 'value', NaN);
        SoloParamHandle(obj, 'sideLickDurOrig', 'value', NaN);
        SoloParamHandle(obj, 'sideLickDurStepOrig', 'value', NaN);
        SoloParamHandle(obj, 'vis_propHMEOrig', 'value', NaN);
        SoloParamHandle(obj, 'aud_propHMEOrig', 'value', NaN);
        SoloParamHandle(obj, 'animalWorking_waitd', 'value', 1);
        SoloParamHandle(obj, 'animalWorking_sideld', 'value', 1);
        SoloParamHandle(obj, 'adaptiveDursOrig', 'value', NaN);
        SoloParamHandle(obj, 'countBad', 'value', 0);
        SoloParamHandle(obj, 'stimdurdiff_reset', 'value', 0);
        SoloParamHandle(obj, 'StimDur_diffOrig', 'value', NaN);
        SoloParamHandle(obj, 'extraStimDur_working', 'value', 1);
        
        % Anti-bias option
        EditParam(obj, 'antiBiasStrength', 0, xpos, ypos); next_row(ypos);
        EditParam(obj, 'AntiBiasTau', 4, xpos, ypos); next_row(ypos);
        SoloParamHandle(obj, 'AntiBiasPrevLR', 'value', NaN);
        SoloParamHandle(obj, 'AntiBiasPrevSuccess', 'value', NaN);
        SoloParamHandle(obj, 'PreviousPropLeft', 'value', 0.5);
        
        % Initialize anti-bias arrays
        % successArray will be: prevTrialLR x prevTrialSuccessful x thisTrialLR
        SoloParamHandle(obj, 'SuccessArray', 'value', 0.5 * ones(2, 2, 2));
        SoloParamHandle(obj, 'ModeRightArray', 'value', 0.5 * ones(1, 3));
      
        % 
        EditParam(obj, 'startScopeDur', 0.5, xpos, ypos); next_row(ypos);
        EditParam(obj, 'stopScopeDur', 0.5, xpos, ypos); next_row(ypos);
        
        % Set the left, right trial array.
        max_trials = 5000;
        left_or_right = 'lr';
        coin = round(rand(1,max_trials));        
        
        % FN: make sure there are no more than th_consectuive trials of the same type in a row
        th_consecutive = 0; %5 % set to 0 if you want the pure random order.
        if th_consecutive > 0
            coin_th = trSeq(coin, th_consecutive);
            coin = coin_th;
        end
        
        sides_list = left_or_right(1 + coin);
        sides_list_bool = coin;
        
        SoloParamHandle(obj, 'SidesList', 'value', sides_list);
%         SidesPlotSection(obj, 'init', 0, 0, value(SidesList));
        SoloParamHandle(obj, 'SidesListBool', 'value', sides_list_bool);
        
        SoloParamHandle(obj, 'HitHistory','value', nan(1, max_trials));
%         SoloParamHandle(obj, 'WasToneFlash', 'value', 0);
        
        SoloParamHandle(obj, 'REncoderSerial', 'value', []);
        
        %% Sides plot and Performance plot
        SidesPlotSection(obj, 'init', 0, 0, value(SidesList));
        PerformancePlotSection(obj, 'init', 0, 0, max_trials,{'left correct','right correct','didNotChoose','valid (notEarlyDecision)', 'wrong initiation'}); % Farzaneh added total and notEarlyWithdrawal. % total correct used to be there instead of didNotChoose
        
        
        %% Pokes plot
        xpos = 10; ypos = 5;
        my_state_colors = struct( ...
            'wait_for_initiation',      [0 0 1], ...
            'wait_for_initiation2',      [0 0 .9], ...            
            'stim_delay',               [0.9655    0.6207    0.8621], ...               
            'wait_stim',                [0 1 0], ...
            'lickcenter_again',         [.85 1 0], ... 
            'center_reward',            [.5 .2 .6], ...
            'wait_for_decision',        [1 0 0], ...
            'wait_for_decision2',       [.9 0 0], ...
            'correctlick_again_wait',   [0 1 0],...
            'correctlick_again',        [0.9655    0.6207    0.8621],...
            'errorlick_again_wait',     [.5 .2 .6],...
            'errorlick_again',          [0.9655    0.6207    0.8621],...
            'direct_water',             [1.0000    0.1034    0.7241], ...
            'reward',                   [1.0000    0.8276         0], ...
            'stopstim_pre',             [1    1    0],...
            'punish_allowcorrection',   [0 .4 0],...
            'punish_timeout',           [0    0.3448         0], ...
            'direct_correct',           [0.5172    0.5172    1.0000], ...  
            'did_not_choose',           [0.6207    0.3103    0.2759], ...
            'wrong_initiation',         [0    1.0000    0.7586], ...  
            'did_not_lickagain',        [0 1 0], ...
            'iti',                      [0    0.5172    0.5862], ...
            'iti2',                     [0    0.5172    0.4], ...
            'check_next_trial_ready',   [0         0    0.4828], ...
            'state_0',                  [0.5862    0.8276    0.3103], ...
            'early_decision',           [0.9655    0.6207    0.8621], ...
            'early_decision0',          [0.8276    0.0690    1.0000]);
%             'last_state',               [0.9655    0.6207    0.8621]);
            
            
        
        my_poke_colors = struct( ...
            'L',    0.6*[1 0.66 0], ...
            'C',    [0 0 0], ...
            'R',    0.9*[1 0.66 0], ...
            'S',    [.5 .5 .5]);
        
        [x, y] = PokesPlotSection(obj, 'init', xpos, ypos, ...
            struct('states', my_state_colors, 'pokes', my_poke_colors));
        
        PokesPlotSection(obj, 'set_alignon', 'stim_delay(1,1)');        
        % gui ends here
        
        %% declare new sounds to be created later
        SoundManagerSection(obj, 'init');
        SoundManagerSection(obj, 'declare_new_sound', 'TargetSound', [0]); % FN changed 'HighOrLowSound' to 'TargetSound'.
%         SoundManagerSection(obj, 'declare_new_sound', 'ExtraTargetSound', [0]); % FN changed 'ExtraHighOrLowSound' to 'ExtraTargetSound'.
        SoundManagerSection(obj, 'declare_new_sound', 'PunishNoise1', [0]);
        SoundManagerSection(obj, 'declare_new_sound', 'PunishNoise2', [0]);
        SoundManagerSection(obj, 'declare_new_sound', 'WaitStart', [0]);
        SoundManagerSection(obj, 'declare_new_sound', 'WaitEnd', [0]);
        
        % The value of RecordingSession will always be zero at this point
        % connect to neuralynx computer
        %{
     if value(RecordingSession) == 1
         reply = NlxConnectToServer('192.168.1.104'); % 192.168.1.104 is the ip of the neuralynx computer in rig 3
         if ~reply
             fprintf('** Could NOT connect to neuralynx computer!');
         end
     end
        %}        
        % target_sound_id = SoundManagerSection(obj, 'get_sound_id', 'TargetSound');
        
        
        %% rotary encoder and arduino
        
        % Close old instruments
        if ~isempty(instrfind)
            fclose(instrfind);
            delete(instrfind);
        end
        
        
        % Initialize rotary encoder's Arduino
        REncoderSerial.value = ArduinoSection_FN(obj, [], 'init_serial_connection');
        
        
        %%
        sma = StateMachineAssembler('full_trial_structure');
        
        sma = add_state(sma, 'name', 'start', ...
            'self_timer', 2, ...
            'output_actions', { 'DOut', lled + cled + rled; }, ...
            'default_statechange', 'check_next_trial_ready');
        
        dispatcher('send_assembler', sma, 'start');        
        
        %---------------------------------------------------------------
        %          CASE PREPARE_NEXT_TRIAL
        %---------------------------------------------------------------
    case 'prepare_next_trial'
        
        my_parsed_events = disassemble(current_assembler, raw_events, 'parsed_structure', 1);
                
        side_label = ['L','R']; % you will need this in 2 places further down, at least...        
        TrialNum.value = value(n_done_trials);
                
        % Farzaneh: related to adaptiveDur: whenever the animal is "working" save the current duration and step values for access in future.
        if value(animalWorking_waitd)
            waitDurOrig.value = value(WaitDuration);
            waitDurStepOrig.value = value(WaitDurationStep);
            
            adaptiveDursOrig.value = value(adaptiveDurs);
            
            vis_propHMEOrig.value = value(VisStrengthPropHME);
            aud_propHMEOrig.value = value(AudStrengthPropHME);
        end
        
        if value(animalWorking_sideld)
            sideLickDurOrig.value = value(SideLickDur);
            sideLickDurStepOrig.value = value(SideLickDurStep);
        end

%         if n_done_trials <= 2 % you commented this out after you added maxInitialBad option (ie warmup for waitdur).
%         max_past_bad = 2; % 2 % you may want to play with this value (how many trials they need to ruin in order for resetting the duration values).
%         if n_done_trials > max_past_bad+1 % (+1 is bc at the beginning a dummy trial is given). you want to be more conservative at the beginning.
            max_past_bad = value(maxPastBad); % value in the gui
%         end
        

        % if waitdur<.15 the LR and HR stimuli wont be easily separable. to avoid this you create the stimulus based on a 1sec stimulus for waitdur<.15.
        if ~value(stimdurdiff_reset)
            StimDur_diffOrig.value = value(StimDur_diff);
        end
        
        
        % do a warmup for extraStimDur: at the beginning set it long enough
        % so mouse can see the stimulus until he makes a decision.
        % also do a warmup for percentAllow: set it to 1 at the beginning.
%         if n_done_trials <= 10
%             ExtraStimDuration.value = 10;
%             PercentAllow.value = 1;
%             extraStimDur_working.value = 0;
%         elseif n_done_trials == 11
%             ExtraStimDuration.value = value(ExtraStimDurOrig);
%             PercentAllow.value = value(PercentAllowOrig);
%         end
        
        if value(extraStimDur_working) && n_done_trials > 10
            ExtraStimDurOrig.value = value(ExtraStimDuration);
        end
        
        
        
%         if (n_done_trials > 1) && ~isempty(my_parsed_events.states.give_center_unconst) % if you used center unconstraned on the next trial set its exclude to 1.
%             Exclude.value = 1;
%         end        

        if value(Exclude)
            if n_done_trials == 2 
                triExcludeStarted.value = 1;            
            elseif n_done_trials > 2 && AllData(n_done_trials - 2).exclude==0
                triExcludeStarted.value = n_done_trials-1;
            end
            
            if n_done_trials - value(triExcludeStarted) == 3+1 % if you want to set exclude to 0 after 5 trials which has been 1.
                Exclude.value = 0;
            end
        end
        
        if value(Exclude) % && value(WaitdurWarmup_noMore)
            adaptiveDurs_force.value = 1;
        else
            adaptiveDurs_force.value = 0;
        end
        
        
        
        % connect to neuralynx computer
        %{
    if value(RecordingSession) == 1 & value(CheckedRecordingSessionValue) == 0
        reply = NlxConnectToServer('192.168.1.104'); % 192.168.1.104 is the ip of the neuralynx computer in rig 3
        if ~reply
            fprintf('** Could NOT connect to neuralynx computer!');
        end
        CheckedRecordingSessionValue.value = 1;
    end
        %}
        
        % send msg to neuralynx computer with the trial id
        %{
    if value(RecordingSession)
        trial_id_msg = sprintf('-PostEvent "this trial id" %d 1', value(TrialNum));
        reply = NlxSendCommand(trial_id_msg);
        if ~reply
            sprintf('** Could NOT send msg to neuralynx computer!');
        end
    end
        %}
        
        if n_done_trials == 1 
            filename = strcat('C:\ratter\SoloData\Data\david\dr_new_mice\', value(SubjectName), ...
                '_', datestr(clock,1), '_', datestr(clock,'HH'), datestr(clock, 'MM'), '.mat');
            shortname = [value(SubjectName), '_', datestr(clock,1), '_', datestr(clock,'HH'), datestr(clock, 'MM')];
            SoloParamHandle(obj, 'FileName', 'value', filename);
            SoloParamHandle(obj, 'ShortFileName', 'value', shortname);  
            
            playStimOrig.value = value(PlayStimulus);
            ExtraStimDurOrig.value = value(ExtraStimDuration);
            PercentAllowOrig.value = value(PercentAllow);
        end


        %% set some parameters for the previous trial.
        
        outcome = NaN;        
        if (n_done_trials > 1)

            AllData(n_done_trials - 1).parsedEvents = my_parsed_events;
            
            %% Sanity checks on duration values
            % in case you have by mistake entered insane values, use the
            % values of the previous trial.
            
            if length(value(WaitDuration))~=1 || value(WaitDuration) > 5 
                WaitDuration.value = AllData(n_done_trials - 1).waitDuration;
            end
            
            if length(value(SideLickDur))~=1 || value(SideLickDur) > 5 
                SideLickDur.value = AllData(n_done_trials - 1).sideLickDur;
            end
            
            if length(value(StimDur_aftRew))~=1
                StimDur_aftRew.value = AllData(n_done_trials - 1).stimDur_aftRew;
            end

            if length(value(ExtraStimDuration))~=1
                ExtraStimDuration.value = AllData(n_done_trials - 1).extraStimDuration;
            end
                    
            
            %% Set the responseSideName of the previous trial.
            %{
            if ~isempty(my_parsed_events.states.flash_detected)
                AllData(n_done_trials - 1).visualStimulusDetected = 1;
            else
                AllData(n_done_trials - 1).visualStimulusDetected = 0;
            end
            %}            
            % (fn attention) are these correct for both allow correction
            % and choose side????
            if ~isempty(my_parsed_events.states.direct_correct) % direct stage
                if ~isempty(my_parsed_events.pokes.L) && ismember(0, my_parsed_events.pokes.L(:)-my_parsed_events.states.wait_for_decision(2))
                    AllData(n_done_trials - 1).responseSideIndex = 1; % Animal chose left

                elseif ~isempty(my_parsed_events.pokes.R) && ismember(0, my_parsed_events.pokes.R(:)-my_parsed_events.states.wait_for_decision(2))
                    AllData(n_done_trials - 1).responseSideIndex = 2; % Animal chose right
                end
                
            elseif ~isempty(my_parsed_events.states.reward)
                if ~isempty(my_parsed_events.pokes.L) && ismember(0, my_parsed_events.pokes.L(:)-my_parsed_events.states.reward(1))
                    AllData(n_done_trials - 1).responseSideIndex = 1; % Animal chose left

                elseif ~isempty(my_parsed_events.pokes.R) && ismember(0, my_parsed_events.pokes.R(:)-my_parsed_events.states.reward(1))
                    AllData(n_done_trials - 1).responseSideIndex = 2; % Animal chose right
                end
                
            elseif ~isempty(my_parsed_events.states.punish)
                if ~isempty(my_parsed_events.pokes.L) && ismember(0, my_parsed_events.pokes.L(:)-my_parsed_events.states.punish(1))
                    AllData(n_done_trials - 1).responseSideIndex = 1; % Animal chose left

                elseif ~isempty(my_parsed_events.pokes.R) && ismember(0, my_parsed_events.pokes.R(:)-my_parsed_events.states.punish(1))
                    AllData(n_done_trials - 1).responseSideIndex = 2; % Animal chose right
                end
                
            elseif isempty(my_parsed_events.states.reward) && ~isempty(my_parsed_events.states.punish_allowcorrection_done)
                if ~isempty(my_parsed_events.pokes.L) && ismember(0, my_parsed_events.pokes.L(:)-my_parsed_events.states.punish_allowcorrection(end,1))
                    AllData(n_done_trials - 1).responseSideIndex = 1; % Animal chose left

                elseif ~isempty(my_parsed_events.pokes.R) && ismember(0, my_parsed_events.pokes.R(:)-my_parsed_events.states.punish_allowcorrection(end,1))
                    AllData(n_done_trials - 1).responseSideIndex = 2; % Animal chose right
                end  
                
            else
                AllData(n_done_trials - 1).responseSideIndex = NaN; % wrong_initiation, early_decision, did_not_lickagain, did_not_choose, did_not_sidelickagain 
            end
%             disp(AllData(n_done_trials - 1).responseSideIndex)
            

            
            if ~isnan(AllData(n_done_trials - 1).responseSideIndex)
                AllData(n_done_trials - 1).responseSideName = side_label(AllData(n_done_trials - 1).responseSideIndex);
            else
                AllData(n_done_trials - 1).responseSideName = '-';
            end
            
%             disp(AllData(n_done_trials - 1).responseSideName)
            
            %{
            % Farzaneh: remember you changed the time units in AllData to sec,  so it matches parsedEvents. 
            time_in_center = 0;
            if ~isempty(my_parsed_events.pokes.C)
                if ~isnan(my_parsed_events.pokes.C(1))
                    time_in_center = diff(my_parsed_events.pokes.C(1,:));
                else
                    if size(my_parsed_events.pokes.C, 1) > 1
                        time_in_center = diff(my_parsed_events.pokes.C(2,:));
                    end
                end
            end
            AllData(n_done_trials - 1).timeInCenter = time_in_center * 1000;
            
            
            movement_duration = nan;
            if ~isempty(my_parsed_events.states.reward)
                movement_duration = my_parsed_events.states.reward(1) - my_parsed_events.states.allowed_to_withdraw(2);
                
            elseif ~isempty(my_parsed_events.states.punish)
                movement_duration = my_parsed_events.states.punish(1) - my_parsed_events.states.allowed_to_withdraw(2);
            end
            AllData(n_done_trials - 1).movementDuration = movement_duration * 1000;
            %}            
            
            %% Set AllData for the outcome of previous trial
            
            % REWARD
            if ~isempty(my_parsed_events.states.reward) || ~isempty(my_parsed_events.states.direct_correct)  
%                 disp('REWARD')
                outcome = 1;                
                HitHistory(n_done_trials-1) = 1; % Correct
                TotalCorrect.value = value(TotalCorrect) + 1;
            end
            
            
            % PUNISH
            if(~isempty(my_parsed_events.states.punish)) || ~isempty(my_parsed_events.states.punish_allowcorrection_done)
%                 disp('PUNISH')
                outcome = 0;
                HitHistory(n_done_trials-1) = 0; % Incorrect
                TotalIncorrect.value = value(TotalIncorrect) + 1;
            end
            
            
            % EARLY DECISION: -1
            if(~isempty(my_parsed_events.states.early_decision)) % this happens when the mouse side licks during waitdur (including pre and post stim delays) or during lickcenter_again (ie when he must center lick one more time).
%                 disp('EARLY DECISION')
                outcome = -1;
                HitHistory(n_done_trials-1) = -1; % early decision
                TotalEarlyDecisions.value = value(TotalEarlyDecisions) + 1;
            end

            
            % DID NOT CHOOSE: -2
            if(~isempty(my_parsed_events.states.did_not_choose))
%                 disp('DID NOT CHOOSE')
                outcome = -2;
                HitHistory(n_done_trials-1) = 2; % did  not choose % it is set to +2 (and not -2 like outcome) bc in performance plot hitHistory >= 0 is counted valid.
                TotalDidNotChoose.value = value(TotalDidNotChoose) + 1;
            end
            
            
            % DID NOT SIDE LICK AGAIN: -5
            if(~isempty(my_parsed_events.states.did_not_sidelickagain))
%                 disp('DID NOT SIDE LICK AGAIN')
                outcome = -5;
                HitHistory(n_done_trials-1) = 3; % did not sidelick again % it is set to +3 (and not -2 like outcome) bc in performance plot hitHistory >= 0 is counted valid.
                TotalDidNotChoose.value = value(TotalDidNotChoose) + 1;
            end            

            
            % WRONG INITIATION : -3
            if(~isempty(my_parsed_events.states.wrong_initiation)) % this state wont be empty if during wait_for_initiation mouse side licks.
%                 disp('WRONG INITIATION')
                outcome = -3;
                HitHistory(n_done_trials-1) = -3; 
                TotalDidNotInitiate.value = value(TotalDidNotInitiate) + 1;
            end

            
            % DID NOT LICK AGAIN: -4
            if(~isempty(my_parsed_events.states.did_not_lickagain))
%                 disp('DID NOT LICK AGAIN')
                outcome = -4;
                HitHistory(n_done_trials-1) = -4; 
                TotalDidNotLickAgain.value = value(TotalDidNotLickAgain) + 1;
            end
            
            AllData(n_done_trials - 1).wrongInitiation = (outcome == -3);
            AllData(n_done_trials - 1).didNotLickagain = (outcome == -4);            
            AllData(n_done_trials - 1).earlyDecision = (outcome == -1);
            AllData(n_done_trials - 1).didNotChoose = (outcome == -2);
            AllData(n_done_trials - 1).didNotSideLickAgain = (outcome == -5);
            AllData(n_done_trials - 1).success = (outcome > 0);
            
            AllData(n_done_trials - 1).outcome = outcome;
            AllData(n_done_trials - 1).exclude = value(Exclude);
            
            %% wheel position from rotary encoder
            
            % Get wheel position trace from rotary encoder
            AllData(n_done_trials - 1).wheelRev = ArduinoSection_FN(obj, value(REncoderSerial), 'pull_traces') * 1/rotaryResolution; % wheel position in units of revolutions. * 455   / rotaryResolution; % 455mm is the periphary of the wheel.
            AllData(n_done_trials - 1).wheelSampleInt = rotarySampleInt;
            AllData(n_done_trials - 1).wheelPostTrialToRecord = postTrialToRecord;
            
            
            %% Water received
            WaterReceived.value = value(TotalCorrect) * value(RewardVolumeSide)/1000 + ...
                (value(TotalCorrect)+value(TotalIncorrect)) * value(RewardVolumeCenter)/1000;            
            
            
            %% if WaitdurWarmup is 1, at the beginning of a session do a warmup for waitdur (after maxInitialBad trials).
            max_initial_bad = value(maxInitialBad);

            if value(adaptiveDurs_force) || (value(WaitdurWarmup) && ~value(WaitdurWarmup_noMore))
                % remember during this time (since trial num is lower than
                % numWarmUp, propHME is [0 0 0 1] regardless of the GUI
                % value.
                    
                if (n_done_trials > max_initial_bad)
                    badOutcomes = [-3, -1, -4]; % wrongInitiation, earlyDecision, didNotLickAgain, % didNotChoose, didNotSideLickAgain % add 0 if you want to also include punishment.
                    pastBad = ismember(HitHistory(n_done_trials - max_initial_bad : n_done_trials-1), badOutcomes);
                end
                
                if value(adaptiveDurs_force) 
                    if (n_done_trials-1 == value(triExcludeStarted)) || ((n_done_trials > max_initial_bad) && (sum(pastBad) == max_initial_bad)) % reset waitDur only on the first trial that was set to exclude not on the next 5 trials that you keep exclude on 1. Or reset it if the mouse does an early deicion.
                        animalWorking_waitd.value = 0;
                        WaitDuration.value = max(.01, value(WaitDuration)*.1); % *.4 % .15 % max(.15, value(WaitDuration)/2); % .15;
                        adaptiveDurs.value = 0; % you don't want to reset durs during waitdur warmup trials.
                        ExtraStimDuration.value = 10;
                        extraStimDur_working.value = 0;
                    end
                    
                elseif (n_done_trials <= 10) && (n_done_trials > max_initial_bad) % if you want the warmup to happen only if the outcome was bad at the very beginning change this to (n_done_trials == max_initial_bad+1).
                    
%                     badOutcomes = [-3, -1, -4]; % wrongInitiation, earlyDecision, didNotLickAgain, % didNotChoose, didNotSideLickAgain % add 0 if you want to also include punishment.
%                     pastBad = ismember(HitHistory(n_done_trials - max_initial_bad : n_done_trials-1), badOutcomes);
                    
                    if sum(pastBad) == max_initial_bad                        
                        animalWorking_waitd.value = 0;
                        WaitDuration.value = max(.01, value(WaitDuration)*.1); % *.4 % .15 % max(.15, value(WaitDuration)/2); % .15;
                        adaptiveDurs.value = 0; % you don't want to reset durs during waitdur warmup trials.
                        ExtraStimDuration.value = 10;
                        extraStimDur_working.value = 0;
                    end
                end
            end

            

            
            % Farzaneh: this part needs work. bc of the following if condition
            % (abs(value(WaitDuration) - value(waitDurOrig)) <= .05), the following will happen: WaitdurWarmup_noMore.value = 1;
            % and hence waitdurWarmup will never take effect!
            if value(adaptiveDurs_force) || (value(WaitdurWarmup) && ~value(WaitdurWarmup_noMore))                
                if ~value(adaptiveDurs)
                    if abs(value(WaitDuration) - value(waitDurOrig)) <= .05 % use the original (before resetting) step values and set orig values to NaN, so later on the GUI values for the step will be used.
                        
                        animalWorking_waitd.value = 1;
                        WaitdurWarmup_noMore.value = 1;
                        
                        WaitDuration.value = value(waitDurOrig); % when waitdur gets within 50ms of the original value go back to the original values.
                        WaitDurationStep.value = value(waitDurStepOrig);
                        adaptiveDurs.value = value(adaptiveDursOrig);
                        adaptiveDurs_force.value = 0;
                        extraStimDur_working.value = 1;
                        ExtraStimDuration.value = value(ExtraStimDurOrig);
                        
                    elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .2                        
                        WaitDurationStep.value = .05;
                        extraStimDur_working.value = 1;
                        ExtraStimDuration.value = value(ExtraStimDurOrig);
                        
                    else % >.2                        
                        WaitDurationStep.value = .1;
%                         Exclude.value = 1; % exclude this trial from off line analysis. (or maybe check for waitDurStep and if it is above a certain value then exclude those trials.)
                    end
                end
                
            end

            if n_done_trials > 10 % after trial 10 you want to use the adaptiveDurs regime to reset durations. before that you want to use the adaptiveDur_warmup regime.
                adaptiveDurs.value = value(adaptiveDursOrig);
            end
            
            
            %% if adaptaiveDurs, reset duration values after consecutive bad trials and update step values accordingly.
            
            if value(adaptiveDurs) && n_done_trials > max_past_bad 
                % adaptively change wait and sideLick duraiton values if there are
                % MaxPastBad consecutive bad trials, otherwise keep using the GUI values.

                badOutcomes = [-3, -1, -4]; % wrongInitiation, earlyDecision, didNotLickAgain, % didNotChoose, didNotSideLickAgain % add 0 if you want to also include punishment.
                pastBad = ismember(HitHistory(n_done_trials - max_past_bad : n_done_trials-1), badOutcomes);

                % Reset waitDur and sideLickDur if there are
                % MaxPastBad consecutive bad trials, otherwise keep using the GUI values.
                if sum(pastBad) == max_past_bad
                    
                    animalWorking_waitd.value = 0;
%                     animalWorking_sideld.value = 0;

                    % rest (considerably lower) the duration values but increase step values.
%                     WaitDuration.value = max(.01, value(waitDurOrig)*.3); % max(.01, value(WaitDuration)*.5);% max(.15, value(WaitDuration)/2); % .15; 
                    WaitDuration.value = max(.01, value(WaitDuration)*.5);
%                     SideLickDur.value = max(.01, value(SideLickDur) - .05); %sideLickDurOrig
                    
                    VisStrengthPropHME.value = [0 0 0 1];
                    AudStrengthPropHME.value = [0 0 0 1];
                    
                    ExtraStimDuration.value = 10;
                    extraStimDur_working.value = 0;
                    
                    %{
                    allbad = (ismember(HitHistory(max(1, n_done_trials-10) : n_done_trials-1), badOutcomes)); % 1:ndone-1
                    countBad.value = floor(value(countBad) + (sum(allbad)/max_past_bad)); % fn attention
                    
                    if value(countBad) >= 3 % if 3 times or more in a session waitdur was reset, then animals seems to be unable to catch up with the waitdur, so decrease the original value.
                        waitDurOrig.value = value(waitDurOrig)*.8;
                    end
                    %}
                end
                   

                %{
                % reset waitdurOrig % fn attention: if after moving the
                % window 1 unit valid rate changes by some amount you want
                % to use the difference with the previous valid rate (of
                % the previous window) and then update durorig based on
                % that difference. so you need to keep nowvalid for each
                % window.
                if n_done_trials > value(NumWarmUpTrials)+20 % compute nowvalid (current valid rate) over the past 20 trials (excluding the warmup trials).
                    nowvalid = nanmean(ismember(HitHistory(max(1, n_done_trials-20) : n_done_trials-1), badOutcomes));
                    goodvalid = .8;
                    diffvalid = (nowvalid - goodvalid)/4;
                    if diffvalid < 0 % if nowvalid is .75, then its difference from defined goodvalid is -.1, hence lower current waitDurOrig by 10%
                        waitDurOrig.value = value(waitDurOrig) + diffvalid * value(waitDurOrig);
                    end
                end
                %}
                
                %%%%%%%%%% update duration step values:
                
                % waitDurStep: gradually update as waitDur gets close to the
                % original value (ie the value before change), when there
                % has been consecutive bad trials.
                if ~value(animalWorking_waitd) % if 0, there has never been consecutive bad trials, or if there has been some in the past, the animal has already gone back to its original value of waitDur.
                    if abs(value(WaitDuration) - value(waitDurOrig)) <= .005 % use the original (before resetting) step values and set orig values to NaN, so later on the GUI values for the step will be used.
                        
                        animalWorking_waitd.value = 1;
%                         countBad.value = value(countBad) - 1;
                        
                        WaitDuration.value = value(waitDurOrig);
                        WaitDurationStep.value = value(waitDurStepOrig);
                        
                        VisStrengthPropHME.value = value(vis_propHMEOrig);
                        AudStrengthPropHME.value = value(aud_propHMEOrig);
                        extraStimDur_working.value = 1;
                        ExtraStimDuration.value = value(ExtraStimDurOrig);
                        
%                     elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .01
%                         WaitDurationStep.value = .0005;
%                         
%                         VisStrengthPropHME.value = value(vis_propHMEOrig);
%                         AudStrengthPropHME.value = value(aud_propHMEOrig);
%                         
%                     elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .015 %.02
%                         WaitDurationStep.value = .001;
%                         
%                         VisStrengthPropHME.value = value(vis_propHMEOrig);
%                         AudStrengthPropHME.value = value(aud_propHMEOrig);
                        
                    elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .02 % .05 % .03 % .06 .07
                        WaitDurationStep.value = .005;
                        
%                         if value(vis_propHMEOrig(2))~= 0 % value(vis_propHMEOrig)~= [0 0 0 1]
%                             VisStrengthPropHME.value = [0 .3 .3 .4];
%                             AudStrengthPropHME.value = [0 .3 .3 .4];
%                         end
                        
                        VisStrengthPropHME.value = value(vis_propHMEOrig);
                        AudStrengthPropHME.value = value(aud_propHMEOrig);
                        extraStimDur_working.value = 1;
                        ExtraStimDuration.value = value(ExtraStimDurOrig);
                        
%                     elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .09 
%                         WaitDurationStep.value = .01;
%                         
%                         if value(vis_propHMEOrig(2))~= 0 % value(vis_propHMEOrig)~= [0 0 0 1]
%                             VisStrengthPropHME.value = [0 .3 .3 .4];
%                             AudStrengthPropHME.value = [0 .3 .3 .4];
%                         end
                        
                    elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .1 % .12 % .15
                        WaitDurationStep.value = .015;
                        
                        if value(vis_propHMEOrig(2))~= 0 % value(vis_propHMEOrig)~= [0 0 0 1]
                            VisStrengthPropHME.value = [0 .3 .3 .4];
                            AudStrengthPropHME.value = [0 .3 .3 .4];
                        end
                        extraStimDur_working.value = 1;
                        ExtraStimDuration.value = value(ExtraStimDurOrig);
                        
                    elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .15
                        WaitDurationStep.value = .02; % you may want to do it more gradual.
                        
                        if value(vis_propHMEOrig(3))~= 0 % value(vis_propHMEOrig)~= [0 0 0 1]
                            VisStrengthPropHME.value = [0 0 .5 .5];
                            AudStrengthPropHME.value = [0 0 .5 .5];
                        end
                        extraStimDur_working.value = 1;
                        ExtraStimDuration.value = value(ExtraStimDurOrig);
                        
                    elseif abs(value(WaitDuration) - value(waitDurOrig)) <= .3
                        WaitDurationStep.value = .08; % .08; % .06 you may want to do it more gradual.
                        
                        VisStrengthPropHME.value = [0 0 0 1];
                        AudStrengthPropHME.value = [0 0 0 1];
                        
                        extraStimDur_working.value = 1;
                        ExtraStimDuration.value = value(ExtraStimDurOrig);
                        
                    else % > .3
                        WaitDurationStep.value = .1; % .08; % .06 you may want to do it more gradual.
                        
                        VisStrengthPropHME.value = [0 0 0 1];
                        AudStrengthPropHME.value = [0 0 0 1];
                        
%                         Exclude.value = 1; % exclude this trial from off line analysis. (or maybe check for waitDurStep and if it is above a certain value then exclude those trials.)

                    end
                end

                
                % sideLickDur
                
                badOutcomes = [2, 3]; % didNotChoose, didNotSideLickAgain % add 0 if you want to also include punishment.
                pastBad = ismember(HitHistory(n_done_trials - max_past_bad : n_done_trials-1), badOutcomes);

                % Reset sideLickDur if there are MaxPastBad consecutive bad trials, otherwise keep using the GUI values.
                if sum(pastBad) == max_past_bad
                    
%                     animalWorking_waitd.value = 0;
                    animalWorking_sideld.value = 0;

                    % rest (considerably lower) the duration values but increase step values.
%                     WaitDuration.value = max(.01, value(WaitDuration)*.5); % max(.15, value(WaitDuration)/2); % .15; 
                    SideLickDur.value = max(.01, value(SideLickDur) - .09); %-.05 %sideLickDurOrig
                    
%                     VisStrengthPropHME.value = [0 0 0 1];
%                     AudStrengthPropHME.value = [0 0 0 1];                                        
                end                
                
                
                % sideLickDurStep: gradually update as sideLickDur gets close to the
                % original value (ie the value before change), when there
                % has been consecutive bad trials.
                if ~value(animalWorking_sideld) % if 0there has never been consecutive bad trials, or if there has been some in the past, the animal has already gone back to its original value of sideLickDur.
                    if abs(value(SideLickDur) - value(sideLickDurOrig)) <= .0015 % .002 % use the original (before resetting) step values and set orig values to NaN, so later on the GUI values for the step will be used.
                        
                        animalWorking_sideld.value = 1;
                        
                        SideLickDur.value = value(sideLickDurOrig);
                        SideLickDurStep.value = value(sideLickDurStepOrig);
                        
%                     elseif abs(value(SideLickDur) - value(sideLickDurOrig)) <= .005
%                         SideLickDurStep.value = .0005;
                        
%                     elseif abs(value(SideLickDur) - value(sideLickDurOrig)) <= .01
%                         SideLickDurStep.value = .001;
                        
                    elseif abs(value(SideLickDur) - value(sideLickDurOrig)) <= .02
                        SideLickDurStep.value = .0015;
                        
                    elseif abs(value(SideLickDur) - value(sideLickDurOrig)) <= .05
                        SideLickDurStep.value = .003;
                        
                    else
                        SideLickDurStep.value = .007;
                    end
                    
                end
            end
                
            
            
            %% Update the duration values if previous trial was correct.    
            if ~isempty(my_parsed_events.states.reward) || ~isempty(my_parsed_events.states.direct_correct) % correct                
                % you may want to define 1.1 and .5 as the max values for
                % durs in the gui.... or you may want to use different values.
                WaitDuration.value = min(1, value(WaitDuration) + value(WaitDurationStep)); %1.1
                SideLickDur.value = min(.2801, value(SideLickDur) + value(SideLickDurStep)); % .3 .5               
                ExtraStimDuration.value = max(0, value(ExtraStimDuration) - value(ExtraStimStep)); % Kachi: added Aug 2013 - decreases extra stimulation duration over time                
                StimDur_aftRew.value = max(0, value(StimDur_aftRew) - value(StimDur_aftRewStep)); % Farzaneh added so the stimulus continues until after the reward is given.
            end
            
            
            %% Update anti-bias history variables if needed and possible.
            % Note that we can't use did_not_choose trials. This is because
            % if he simply stops responding, by chance we'll present more
            % of one side or the other and this will snowball until he's
            % only getting lefts (or rights).
            if outcome > -1
              % Update arrays              
              [newModeRightArray, newSuccessArray] = ...
                  updateAntiBiasArrays(value(ModeRightArray), value(SuccessArray), ...
                  AllData(n_done_trials - 1).visualOrAuditory, outcome, ...
                  AllData(n_done_trials - 1).correctSideIndex, ...
                  value(AntiBiasPrevLR), value(AntiBiasPrevSuccess), ...
                  value(AntiBiasTau), AllData(n_done_trials - 1).responseSideIndex - 1);

              % Update history
              AntiBiasPrevLR.value = AllData(n_done_trials - 1).correctSideIndex;
              AntiBiasPrevSuccess.value = AllData(n_done_trials - 1).success;

              % Update matrices
              SuccessArray.value = newSuccessArray;
              ModeRightArray.value = newModeRightArray;
              
            end            
            
        end

        
    
        %% set number of events for each strength group (hard, medium-hard, medium-easy, easy)
        
        if value(TrialNum) < value(NumWarmUpTrials)
            vis_strength_prop = [0 0 0 1];
            aud_strength_prop = [0 0 0 1];
        else            
            vis_strength_prop = value(VisStrengthPropHME);
            aud_strength_prop = value(AudStrengthPropHME);
        end
        
        
        %% Visual or auditory or both
        coin = rand;
        if coin < value(PropOnlyAudio) % only audio
            show_visual = 0;
            show_audio = 1;
            modality = 1;
        elseif coin < value(PropOnlyVisual) + value(PropOnlyAudio) % only visual
            show_audio = 0;
            show_visual = 1;
            modality = 3;
            % disp('*** ONLY VISUAL ***');
        else % audio & visual
            show_visual = 1;
            show_audio = 1;
            modality = 2;
            % disp('*** AUDIO & VISUAL ***');
        end
        
        
        %% Update the sequence of trials if needed for the coming trial
        % Anti-bias for coming trial
        % If we've previously done a trial, are using anti-bias, and the
        % previous trial wasn't an early withdrawal or failure to choose,
        % update the next trial.
        % Note: at this point in the trial, 'outcome' is still from the
        % previous trial
        %         if n_done_trials > 1 && value(PropLeftTrials) ~= value(PreviousPropLeft)
        
        % First ensure that anti-bias strength is a sane value, between 0 and 1
        if value(antiBiasStrength) < 0 || isempty(value(antiBiasStrength))
          antiBiasStrength.value = 0;
        elseif value(antiBiasStrength) > 1
          antiBiasStrength.value = 1;
        end

        if n_done_trials > 1 && value(antiBiasStrength) > 0 && outcome > -1
          pLeft = getAntiBiasPLeft(value(SuccessArray), value(ModeRightArray), modality, ...
              value(antiBiasStrength), value(AntiBiasPrevLR), value(AntiBiasPrevSuccess));
            
          sides_list = value(SidesList);
          sides_list_bool = value(SidesListBool);
          left_or_right = 'lr';
          coin = rand(1);
          
          sides_list(value(TrialNum)) = left_or_right(1 + (coin < 1 - pLeft));
          sides_list_bool(value(TrialNum)) = (coin < 1 - pLeft);
          
          SidesList.value = sides_list;
          SidesListBool.value = sides_list_bool;

        elseif n_done_trials > 1 && value(antiBiasStrength) == 0 && value(PropLeftTrials) ~= value(PreviousPropLeft)
          max_trials = 5000;
          left_or_right = 'lr';
          coin = rand(1, max_trials);
          old_sides_list = value(SidesList);
          old_sides_list_bool = value(SidesListBool);
          
          sides_list = left_or_right(1 + (coin < (1 - value(PropLeftTrials))));
          sides_list(1:value(TrialNum)) = old_sides_list(1:value(TrialNum));
          sides_list_bool = coin < (1 - value(PropLeftTrials));
          sides_list_bool(1:value(TrialNum)) = old_sides_list_bool(1:value(TrialNum));
          PreviousPropLeft.value = value(PropLeftTrials);
          
          SidesList.value = sides_list;
          SidesListBool.value = sides_list_bool;
        end        
        
        %%
        SidesPlotSection(obj, 'update', value(TrialNum), value(SidesList), value(HitHistory));
        PerformancePlotSection(obj, 'update', value(TrialNum), value(HitHistory), value(SidesListBool) == 0);
        
        %%
        PreStimDelay.value = generate_random_exp(value(PreDelayLambda), value(PreDelayMin), value(PreDelayMax));
        PostStimDelay.value = generate_random_exp(value(PostDelayLambda), value(PostDelayMin), value(PostDelayMax));
        ITI.value = generate_random_exp(value(ITI_lambda), value(ITI_min), value(ITI_max));

        %%
        percentAllow = value(PercentAllow);
        if n_done_trials <= value(NumWarmUpTrials)        
            percentAllow = 1;
        end
        
        r = 1; while r==1, r = rand; end
        if strcmp(value(RewardStage), 'Direct')
            rewardStage.value = 'Direct';
        else
            if r < percentAllow
                rewardStage.value = 'Allow correction';
            else
                rewardStage.value = 'Choose side';
            end
        end
        

        %% Once more sanity checks on duration values before they are used for the generation of the stimulus.       
        % in case you have by mistake entered insane values, use the
        % values of the previous trial.        
        if (n_done_trials > 1)
            
            if length(value(WaitDuration))~=1 || value(WaitDuration) > 5 
                WaitDuration.value = AllData(n_done_trials - 1).waitDuration;
            end
            
            if length(value(SideLickDur))~=1 || value(SideLickDur) > 5 
                SideLickDur.value = AllData(n_done_trials - 1).sideLickDur;
            end
            
            if length(value(StimDur_aftRew))~=1
                StimDur_aftRew.value = AllData(n_done_trials - 1).stimDur_aftRew;
            end

            if length(value(ExtraStimDuration))~=1
                ExtraStimDuration.value = AllData(n_done_trials - 1).extraStimDuration;
            end
        end

        
        %%
        wait_dur = value(WaitDuration);
        sidelick_dur = value(SideLickDur);
        stimdur_aftrew = value(StimDur_aftRew);
        extrastim_dur = value(ExtraStimDuration);
        
        
        %%
        if n_done_trials <= 5
            extrastim_dur = 10;
        end
        
        
        %% set number of events (evnum) for the current stimulus for both the
        % low-rate (evnum(1)) and high-rate (evnum(2)) cases.
        
        if wait_dur < .15
            if ~value(StimDur_diff)
                StimDur_diff.value = 1; % if waitdur<.15 the LR and HR stimuli wont be easily separable. to avoid this you create your stimulus based on a 1sec stimulus.
                stimdurdiff_reset.value = 1;
            end
        else
            StimDur_diff.value = value(StimDur_diffOrig);
            stimdurdiff_reset.value = 0;
        end
        
        
        if value(StimDur_diff) % if it is non zero, then use this value instead of waitDur to generate the stimulus.
            stimdur = round(value(StimDur_diff)*1000); % ms
        else % if StimDur_diff is zero, use waitDur to generate the stimulus. 
            stimdur = round(wait_dur*1000); % ms
        end
        
        evdur = value(EventDuration)*1000; % ms
        gapdur = value(MinInterval)*1000; % ms
        max_iei_accepted = value(MaxInterval)*1000; % ms        
        
        if stimdur < 2*(evdur+gapdur) % you want at least 1 event for lo rate and 2 events for hi rate in order to creat the stimulus.
            PlayStimulus.value = 0;
        else
            PlayStimulus.value = value(playStimOrig);
        end
        
        evnum_vis = NaN(1,2);
        evnum_aud = NaN(1,2);
        if value(PlayStimulus)
            if show_visual
                [evnum_vis, hiSall, loSall, min_evnum, max_evnum, categ_bound_num] = set_evnum(stimdur, evdur, gapdur, max_iei_accepted, value(CategoryBoundaryHz), vis_strength_prop);
            end
            
            if show_audio
                [evnum_aud, hiSall, loSall, min_evnum, max_evnum, categ_bound_num] = set_evnum(stimdur, evdur, gapdur, max_iei_accepted, value(CategoryBoundaryHz), aud_strength_prop);
            end
            
            mn_cb_mx.value = [min_evnum, categ_bound_num, max_evnum];
            lo_hi_1.value = sort([loSall{1} hiSall{1}]);
            lo_hi_2.value = sort([loSall{2} hiSall{2}]);
            lo_hi_3.value = sort([loSall{3} hiSall{3}]);
            lo_hi_4.value = sort([loSall{4} hiSall{4}]);
            
        end

        % use the values that you entered in the GUI for the event numbers.
        lh_input = value(lo_hi_input);
        if ~isempty(lh_input)
            loS = lh_input(lh_input < categ_bound_num);
            hiS = lh_input(lh_input > categ_bound_num);
        
            if rand>.5
                loS = [loS, lh_input(lh_input == categ_bound_num)];
            else
                hiS = [hiS, lh_input(lh_input == categ_bound_num)];
            end
            
            % pick 2 random values of evnum (number of events) from loS and hiS (probability
            % that a particular evnum happens is equal for all evnums in strengthGroup)
            vthisS = 0: 1/length(loS) :1;
            r = 1; while any(r==1), r = rand; end
            [n,i] = histc(r,vthisS);
            evnum(1) = loS(i);
            
            vthisS = 0: 1/length(hiS) :1;
            r = 1; while any(r==1), r = rand; end
            [n,i] = histc(r,vthisS);            
            evnum(2) = hiS(i);
            
            if show_visual, evnum_vis = evnum; end
            if show_audio, evnum_aud = evnum; end
        end

        
        %% Define next trial correct/error side
        
        centerLick = 'Cin';
        water_duration_ave = (value(RightWater) + value(LeftWater)) / 2;
        
        if (SidesList(value(TrialNum)) == 'l')
            correctLick = 'Lin';
            errorLick = 'Rin'; 
            CorrectSideName.value = 'LEFT';
            correct_side_index = 1;
            water_delivery = lwater;
            water_duration = value(LeftWater);
            
            if value(PlayStimulus)
                if strcmpi(value(HighRateChoicePort),'R')
                    evnumV = evnum_vis(1);
                    evnumA = evnum_aud(1);
                else
                    evnumV = evnum_vis(2);
                    evnumA = evnum_aud(2);                
                end
                
            else
                evnumV = NaN;
                evnumA = NaN;
            end
            
        else            
            correctLick = 'Rin';
            errorLick = 'Lin';                        
            CorrectSideName.value = 'RIGHT';
            correct_side_index = 2;
            water_delivery = rwater;
            water_duration = value(RightWater);
            
            if value(PlayStimulus)
                if strcmpi(value(HighRateChoicePort),'R')
                    evnumV = evnum_vis(2);
                    evnumA = evnum_aud(2);
                else
                    evnumV = evnum_vis(1);
                    evnumA = evnum_aud(1);                
                end
                
            else
                evnumV = NaN;
                evnumA = NaN;
            end
            
        end
                
%         eventNumber_VA.value = [evnumV evnumA];
%         disp('_________________')
%         disp([evnumV evnumA])
        
        %% generate a poisson stimulus (Farzaneh)
        
        % compute ieis: inter-event intervals.
        ieisV = NaN;
        ieisA = NaN;
        if value(PlayStimulus) 
            if show_visual
                ieisV = set_stim_fn(stimdur, evdur, gapdur, evnumV, max_iei_accepted); % visual 
            end
            
            if show_audio
                ieisA = set_stim_fn(stimdur, evdur, gapdur, evnumA, max_iei_accepted); % auditory
            end
        end        
        
        %%% OLD stimulus (50,100ms interval): Set the stimulus matrix
        %{
        % generate stimulus space; % The output is a matrix with 5 columns:
        % [stim_strength, nShortIntervals, nLongIntervals, totalDuration, nEvents]
        M = create_stim_matrix(value(ShortInterval)*1000, value(LongInterval)*1000, value(EventDuration)*1000, value(TotalStimDuration)*1000);
        
        % randomly pick a possible stimulus with the desired strength / desired nr. of events
        [stim, actual_events, actual_duration] = get_possible_stim(M, desired_events);
        
%         nr_of_events = actual_events; % FN commented, seems not to be used.
        stim_duration = actual_duration;                
        %}
        
        %% Set is_synch and add offset to the stimulus if needed 
        % (Farzaneh revised this part)
        
        is_synch = NaN; % single modality or noPlayStimulus conditions.
        audio_offset = 0;
        visual_offset = 0;
        
        % multisensory: revise ieisV and ieisA and add offset if needed.
        if show_visual && show_audio && value(PlayStimulus) 
            if rand < value(PropSynchMultisensory) % sync
                is_synch = 1;
                ieisV = ieisA; % use the same intervals for both auditory and visual.
                evnumV = evnumA;
                
            else % async
                is_synch = 0;
                offset = rand*value(MinInterval);
%                 stim_duration = stimdur + offset*1000;
                if rand < 0.5
                    visual_offset = 0;
                    audio_offset = offset;
                else
                    visual_offset = offset;
                    audio_offset = 0;
                end
            end    
            
        end        
        
        eventNumber_VA.value = [evnumV evnumA];
        
       % FN check if the two codes below are needed. if not, delete them!
%         NumberEvents.value = actual_events;
%         StimDuration.value = actual_duration;

        
        %% Set srate and SoundCalibrationParameters 
        
        srate = SoundManagerSection(obj, 'get_sample_rate'); %FN moved it here from somewhere in the bottom.
%         target_sound = 'HighOrLowSound';
%         extra_target_sound = 'ExtraHighOrLowSound';
        
        if strfind(value(SoundCarrier), '15 KHz')
            SoundCalibrationParameters = value(ToneLinearModelParams);
        else
            SoundCalibrationParameters = value(WhiteNoiseLinearModelParams);
        end
        
              
        %% Set the stimulus waveform (Farzaneh)
        
        % Farzaneh modified it so it is not the case anymore (ie no extra event at the end): The funcions below add one event at the end of the last inter-event interval (ieis).
        % Therefore the stimulus (signal_visual) will play for evdur ms longer
        % than stimdur (ie the duration of an event/flash).
        
        if value(PlayStimulus)
            if show_audio
                signal_audio = create_audio_signal_fn(ieisA/1000, srate, value(EventDuration), ...
                    value(SoundCarrier), value(SoundLoudness), audio_offset, value(NoiseMaskAmp), SoundCalibrationParameters);
            else
                signal_audio = 0;
            end

            if show_visual
                signal_visual = create_visual_signal_fn(ieisV/1000, srate, value(Brightness), value(EventDuration), visual_offset);
            else
                signal_visual = 0;
            end
        end
        

        %% Add extraStimDuration to the stim waveform if needed. This will generate the final stimulus signal (auditory and visual) which will be sent to the sound card of linux.
        % FN commented Kachi's codes, and rewrote this section. Kachi's code did not have an inter-event
        % interval between the original stimulus and the extraStimulus. Nov 2014.
                
        if value(PlayStimulus)            
            if extrastim_dur + stimdur_aftrew ~= 0
                
                extra_stim_length = round((extrastim_dur + stimdur_aftrew)*srate);
                                
                if show_visual
                    % set the extra stim.
%                     sep = zeros(1, round(srate * ieisV(randi(length(ieisV)))/1000 ));
%                     sep_visual = sep;
                    
                    nRepeats = ceil(extra_stim_length / length(signal_visual));
%                     extra_signal_visual = repmat([sep_visual  signal_visual], [1 nRepeats]);
                    extra_signal_visual = repmat(signal_visual, [1 nRepeats]);
                    extra_signal_visual = extra_signal_visual(1:extra_stim_length);
                    
                    % set the final stim; i.e. append extra stim to the original stim.
                    signal_visual = [signal_visual  extra_signal_visual];                    
                end                    

                if show_audio
                    % set the extra stim.
%                     noise_amplitude = 10^(1/10*(SoundCalibrationParameters(1)*value(NoiseMaskAmp) + SoundCalibrationParameters(2)));
%                     sep = zeros(1, round(srate * ieisA(randi(length(ieisA)))/1000 ));                    
%                     sep = sep + noise_amplitude*2*(rand(size(sep))-0.5); % add masking noise to sep_auditory
%                     sep_auditory = sep;
                    
                    nRepeats = ceil(extra_stim_length / length(signal_audio));
                    extra_signal_audio = repmat(signal_audio, [1 nRepeats]);  % you gotta put sep_audio in the repat (just like visual)... i think you will get concatenated events here. check          
%                     extra_signal_audio = [sep_auditory  extra_signal_audio(1:extra_stim_length)];
                    extra_signal_audio = extra_signal_audio(1:extra_stim_length);
                    
                    % set the final stim; i.e. append extra stim to the original stim.
                    signal_audio = [signal_audio  extra_signal_audio];
                end            
            end
        end
        
            
        % make sure audio and visual signals have the same length.
        if value(PlayStimulus)
            if length(signal_visual) < length(signal_audio)
                signal_visual = [signal_visual, zeros(1,length(signal_audio) - length(signal_visual))];
            elseif length(signal_visual) > length(signal_audio)
                signal_audio = [signal_audio zeros(1, length(signal_visual) - length(signal_audio))];
            end
        end
        
        
        if value(PlayStimulus)
            stim_duration = length(signal_visual)/srate;
        else
            stim_duration = NaN;
        end
        
        %{
        figure; hold on; 
        plot((1:length(signal_audio))/srate, signal_audio)
        plot((1:length(signal_visual))/srate, signal_visual, 'g')
        %}
        
        %% Create the final signal that will be sent to the sound card of linux.
        if value(PlayStimulus)
            SoundManagerSection(obj, 'set_sound', 'TargetSound', [signal_visual; signal_audio]);    
        else
            SoundManagerSection(obj, 'set_sound', 'TargetSound', []);
%             SoundManagerSection(obj, 'set_sound', 'ExtraTargetSound', []);            
        end
        
        
        %% Create punishment noise for incorrect choices. 
        % (In rats this noise is used for early withdrawals).
        % We need 2 sound buffers because when the state machine is preparing
        % the next trial, the withdrawal punishment noise may still be playing.
        
        if mod(n_done_trials, 2) == 0
            pnoise_name = 'PunishNoise2';
        else
            pnoise_name = 'PunishNoise1';
        end
        
        pnoise_loudness = value(PNoiseLoudness);
        pnoise_duration = value(PunishNoiseDuration);
        
        if strcmpi(value(PunishNoise), 'Pure tone')
            PunishNoiseModelParams = value(ToneLinearModelParams);
            pnoise_amplitude = 10^(1/10*(PunishNoiseModelParams(1)*pnoise_loudness + PunishNoiseModelParams(2)));
            freq = 15000; % Hz
            timevec = (0 : 1/srate : pnoise_duration);
            pnoise = pnoise_amplitude * (sin(2*pi * freq * timevec));
        else
            PunishNoiseModelParams = value(WhiteNoiseLinearModelParams);
            pnoise_amplitude = 10^(1/10*(PunishNoiseModelParams(1)*pnoise_loudness + PunishNoiseModelParams(2)));
            % pnoise = 2 * pnoise_loudness * rand(1, pnoise_duration * srate) - pnoise_loudness;
            pnoise = pnoise_amplitude * rand(1, pnoise_duration * srate);
        end        
        
        SoundManagerSection(obj, 'set_sound', pnoise_name, [zeros(1,length(pnoise)); pnoise]);
        
        
        %% Create the Start Tone and the Go Tone
        se_loudness = value(StartGoToneLoudness);
        
        start_end_tones_duration = 0.05;
        timevec = (0 : 1/srate : start_end_tones_duration);
        envelope = sin([0:pi/(srate * 0.01):pi]).^2;
        
        seNoiseModelParams = value(WhiteNoiseLinearModelParams);
        se_tone_amplitude = 10^(1/10*(seNoiseModelParams(1)*se_loudness + seNoiseModelParams(2))); % 70db for the start and go tone (Farzaneh).
        
        % set tone that signals wait start % Trial-Initiation tone
        freq = 3000; % Hz
        waitstart = se_tone_amplitude * (sin(2*pi * freq * timevec));
        missing_points = ones(1, length(waitstart) - length(envelope));
        waitstart = waitstart .* [envelope(1:round(length(envelope)/2)) missing_points envelope(round(length(envelope)/2)+1:end)];
        
        % set tone that signals wait end % Go tone
        freq = 9000; % Hz
        waitend = 8*se_tone_amplitude * (sin(2*pi * freq * timevec)); % 2
        missing_points = ones(1, length(waitend) - length(envelope));
        waitend = waitend .* [envelope(1:round(length(envelope)/2)) missing_points envelope(round(length(envelope)/2)+1:end)];
        
        %updated 27-jan-2014
        if value(PlayGoTone) == 1 && value(PlayStartTone) ==1
            SoundManagerSection(obj, 'set_sound', 'WaitStart', [zeros(1,length(waitstart)) ; waitstart]);
            SoundManagerSection(obj, 'set_sound', 'WaitEnd', [zeros(1,length(waitend)) ; waitend]);
            
        elseif value(PlayGoTone) == 0 && value(PlayStartTone) ==1
            SoundManagerSection(obj, 'set_sound', 'WaitStart', [zeros(1,length(waitstart)) ; waitstart]);
            SoundManagerSection(obj, 'set_sound', 'WaitEnd', []);
            
        elseif value(PlayGoTone) == 1 && value(PlayStartTone) ==0
            SoundManagerSection(obj, 'set_sound', 'WaitEnd', [zeros(1,length(waitend)) ; waitend]);
            SoundManagerSection(obj, 'set_sound', 'WaitStart', []);
            
        else
            SoundManagerSection(obj, 'set_sound', 'WaitStart', []);
            SoundManagerSection(obj, 'set_sound', 'WaitEnd', []);
        end
        
        
        %% Upload sounds to sound server
        SoundManagerSection(obj, 'send_not_yet_uploaded_sounds');
        
        %% set the barcode (interleaved 2of5) representing trial number.
        % You will send this to the scope (as the trial code). 
        code = encode2of5(value(TrialNum));
        
        %{
        % Farzaneh: you may add the following. If not, the 1st 99 trials will have a
        % pre-stim delay that is 23ms shorter than the rest of the trials.
        % bc code dur is 35ms for the 1st 99 trials, and for 100-999 trials
        % it will be 58ms).
        % To make the duration of the barcode the same for all
        % trials, if trial number<100, add the code corresponding to 00 to
        % the beginning of its code).
        if value(TrialNum) < 100
            code00 = [1 1 2 2 1; 1 1 2 2 1];
            code = [code(:,1:2) , code00 , code(:,3:end)];
        end
        %}
        
        durCode = sum(codeModuleDurs(code(:)));
        
        %% Set AllData. 
        AllData(n_done_trials).species = value(Species);
        AllData(n_done_trials).trialId = value(TrialNum);
        
        if n_done_trials >= 1
            AllData(n_done_trials).filename = value(ShortFileName);
        end
        
        AllData(n_done_trials).highRateChoicePort = value(HighRateChoicePort);        
        AllData(n_done_trials).correctSideIndex = correct_side_index; % correctSideIndex = 1 (left), correctSideIndex = 2 (right)
        AllData(n_done_trials).correctSideName = side_label(AllData(n_done_trials).correctSideIndex);
        AllData(n_done_trials).rewardStage = value(rewardStage);        
        AllData(n_done_trials).percentAllow = value(PercentAllow);
        AllData(n_done_trials).animalWorking_waitd = value(animalWorking_waitd);
        AllData(n_done_trials).animalWorking_sideld = value(animalWorking_sideld);
                
%         AllData(n_done_trials).stimWave_va = [signal_visual; signal_audio]; % stimulus waveform that was sent to the sound card for the visual and audio channels.
        AllData(n_done_trials).auditoryIeis = ieisA/1000;
        AllData(n_done_trials).visualIeis = ieisV/1000;
        
        if isnan(ieisA)
            AllData(n_done_trials).nAuditoryEvents = 0;
        else
            AllData(n_done_trials).nAuditoryEvents = length(ieisA);
        end
        
        if isnan(ieisV)
            AllData(n_done_trials).nVisualEvents = 0;
        else
            AllData(n_done_trials).nVisualEvents = length(ieisV);
        end
        
%         AllData(n_done_trials).nAuditoryEvents = length(ieisA); % + 1;
%         AllData(n_done_trials).nVisualEvents = length(ieisV); % + 1;
        
        AllData(n_done_trials).durTrialCode = durCode;
        AllData(n_done_trials).stimDur_diff = value(StimDur_diff);
        AllData(n_done_trials).stimDuration = stim_duration; % this includes the extra stim and stim after reward too. bur remember if the animal made his choice earlier than the end of stim_duration, then the actual stimulus duration is shorter than this value, bc stim gets stopped when the animal makes a choice.
        AllData(n_done_trials).waitDuration = wait_dur;
        AllData(n_done_trials).preStimDelay = value(PreStimDelay);
        AllData(n_done_trials).postStimDelay = value(PostStimDelay);
        AllData(n_done_trials).totalWaitDuration = (wait_dur + value(PreStimDelay) + value(PostStimDelay) + durCode);
        AllData(n_done_trials).waitDurationStep = value(WaitDurationStep);
        AllData(n_done_trials).extraStimDuration = extrastim_dur; % for rats extraStimDuration should be set to 0
        AllData(n_done_trials).extraStimStep = value(ExtraStimStep);
        AllData(n_done_trials).stimDur_aftRew = stimdur_aftrew;
        AllData(n_done_trials).stimDur_aftRewStep = value(StimDur_aftRewStep);
        AllData(n_done_trials).sideLickDur = sidelick_dur;
        AllData(n_done_trials).sideLickDurStep = value(SideLickDurStep);
        
        AllData(n_done_trials).playStimulus = value(PlayStimulus);
        AllData(n_done_trials).playGoTone = value(PlayGoTone);
        AllData(n_done_trials).playStartTone = value(PlayStartTone);
        AllData(n_done_trials).numWarmUp = value(NumWarmUpTrials);     
        AllData(n_done_trials).isRecordingSession = value(RecordingSession);
        AllData(n_done_trials).categoryBoundaryHz = value(CategoryBoundaryHz);
        
        AllData(n_done_trials).visualOrAuditory = show_visual - show_audio; % 1 only visual, 0 visual & audio, -1 only audio
        AllData(n_done_trials).showVisual = show_visual;
        AllData(n_done_trials).showAudio = show_audio;
        AllData(n_done_trials).visualOffset = visual_offset;
        AllData(n_done_trials).audioOffset = audio_offset;        
        AllData(n_done_trials).isSynch = is_synch;

        AllData(n_done_trials).eventDuration = value(EventDuration);
        AllData(n_done_trials).minInterval = value(MinInterval);
        AllData(n_done_trials).maxInterval = value(MaxInterval);
        
        AllData(n_done_trials).propLeftTrials = value(PropLeftTrials);
        AllData(n_done_trials).propOnlyAudio = value(PropOnlyAudio);
        AllData(n_done_trials).propOnlyVisual = value(PropOnlyVisual);          
        AllData(n_done_trials).propSynchMultisens = value(PropSynchMultisensory);
        AllData(n_done_trials).visStrengthPropHME = value(VisStrengthPropHME);
        AllData(n_done_trials).audStrengthPropHME = value(AudStrengthPropHME);
        
        AllData(n_done_trials).iti = value(ITI);
        AllData(n_done_trials).iti_min = value(ITI_min);
        AllData(n_done_trials).iti_max = value(ITI_max);
        AllData(n_done_trials).iti_lambda = value(ITI_lambda);
        
        AllData(n_done_trials).brightness = value(Brightness); % brightness of the visual stimulus
        AllData(n_done_trials).soundCarrier = value(SoundCarrier);
        AllData(n_done_trials).soundLoudness = value(SoundLoudness);
        AllData(n_done_trials).noiseMaskLoudness = value(NoiseMaskAmp);
        AllData(n_done_trials).punishNoise = value(PunishNoise);
        AllData(n_done_trials).punishNoiseLoudness = value(PNoiseLoudness);
        AllData(n_done_trials).punishNoiseDuration = value(PunishNoiseDuration);
        
        AllData(n_done_trials).antiBiasStrength = value(antiBiasStrength);
        AllData(n_done_trials).antiBiasTau = value(AntiBiasTau);        
        
        AllData(n_done_trials).centerAgainDur = value(CenterAgain);
        AllData(n_done_trials).sideAgainDur = value(SideAgain);
        
        AllData(n_done_trials).leftWaterValveTime = value(LeftWater);
        AllData(n_done_trials).rightWaterValveTime = value(RightWater);
        AllData(n_done_trials).centerWaterValveTime = value(CenterWater);
        AllData(n_done_trials).waterReceived = value(WaterReceived);
        AllData(n_done_trials).centerPoke_amount = value(CenterPoke_amount);
        AllData(n_done_trials).centerPoke_when = value(CenterPoke_when);
        AllData(n_done_trials).startScopeDur = value(startScopeDur);
        AllData(n_done_trials).stopScopeDur = value(stopScopeDur);
        
        AllData(n_done_trials).waitDurWarmup = value(WaitdurWarmup); 
        AllData(n_done_trials).adaptiveDurs = value(adaptiveDurs);
        
        
        %% Set the state matrix
        
        if strcmp(value(rewardStage), 'Direct')
            randomState = 'direct_water';
        else
            randomState = 'wait_for_decision';
        end
            
        startTone = SoundManagerSection(obj, 'get_sound_id', 'WaitStart'); % trial-initiation tone
        flickStim = SoundManagerSection(obj, 'get_sound_id', 'TargetSound'); % flickering stimulus, the main stimulus (visual/auditory)
        goTone = SoundManagerSection(obj, 'get_sound_id', 'WaitEnd'); % go tone
        incorrectTone = SoundManagerSection(obj, 'get_sound_id', pnoise_name); % incorrect-choice tone
        
        sma = StateMachineAssembler('full_trial_structure');
        
        sma = add_scheduled_wave(sma, 'name', 'center_unconst', 'preamble', randi(value(CenterPoke_when)));
        
        trial_ports = ardttl1 + scopeTTL;
        
        %% ITI 
        % in your new code (with the start_rotary_scope state), mouse will have to stop licking for 500ms
        % in order to get the start tone. Then it seems the iti, iti2
        % states and preventing the mouse from licking at those states
        % could be redundant.... maybe you can remove the no-lick
        % constraint during iti, iti2 states.
        
        % iti : so ITI indicates ITI from previous trial
%         sma = add_state(sma, 'name', 'iti',...
%             'self_timer', max(1E-6, value(ITI)-1),... % you are subtracting 1sec, bc you are adding 500ms to the ITI right before the start tone, and 500ms right after the final state. However you are preventing the mouse from lick only during the first 500ms, so in this new code the mouse is forced for a shorter amount of time not to lick during iti.
%             'input_to_statechange', {'Tup','start_rotary_scope', centerLick,'iti2', correctLick,'iti2', errorLick,'iti2'});        
        
        sma = add_state(sma, 'name', 'iti',...
            'self_timer', max(1E-6, value(ITI)-1),... % you are subtracting 1sec, bc you are adding 500ms to the ITI right before the start tone, and 500ms right after the final state. However you are preventing the mouse from lick only during the first 500ms, so in this new code the mouse is forced for a shorter amount of time not to lick during iti.
            'input_to_statechange', {'Tup','trial_start_rot_scope', correctLick,'iti2', errorLick,'iti2'});
        
        
        % iti2
%         sma = add_state(sma, 'name', 'iti2',...
%             'self_timer', max(1E-6, value(ITI)-1),...        
%             'input_to_statechange', {'Tup','start_rotary_scope', centerLick,'iti', correctLick,'iti', errorLick,'iti'});    
        
        sma = add_state(sma, 'name', 'iti2',...
            'self_timer', max(1E-6, value(ITI)-1),...        
            'input_to_statechange', {'Tup','trial_start_rot_scope', correctLick,'iti', errorLick,'iti'});
        
        
        %% Start acquiring mscan data and send the trialStart signal (also the rotary signal). 
        % I think you should have the following asap to make the length of
        % trialStart constant in the analogue channel and make computatoins
        % easier!
        %
        sma = add_state(sma, 'name', 'trial_start_rot_scope',...
            'self_timer', 0.036,...        
            'output_actions', {'DOut', trial_ports + trialStart},...
            'input_to_statechange', {'Tup','start_rotary_scope'});
        
        % remove trialStart from start_rotary_scope. add the name of this
        % stateto Tup iti and iti2.
        %
        
        %% change the name of this state when you start new mice! 
        % The name of this state is misleading now; it is really a short
        % nocenterlick state right before trial initiation.
        
        %%% start_rotary_scope --> scopeTTL gets sent here.
        % you set it to 500ms, bc you want to record imaging and rotary data during some part of the ITI as well.
        sma = add_state(sma, 'name', 'start_rotary_scope',...
            'self_timer', value(startScopeDur),... % 0.035 is the min. Matt: Have your protocol wait at least 35 ms between telling the microscope to start scanning and sending the trial code. The scope is always running the scan mirrors, and waits until the start of the next frame to actually start recording. If you don't do this, you'll lose the start of the codes.
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','trialCode1', centerLick,'start_rotary_scope2', correctLick,'start_rotary_scope2', errorLick,'start_rotary_scope2'});
        
        
        sma = add_state(sma, 'name', 'start_rotary_scope2',...
            'self_timer', value(startScopeDur),... 
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','trialCode1', centerLick,'start_rotary_scope', correctLick,'start_rotary_scope', errorLick,'start_rotary_scope'});
        
        
        %% (still ITI) Sending the trial code: barcode 2of5 indicates trial number. This will be sent to an AI line on the scope machine.        
        % right now you are allowing the mouse to lick during trialCode time (even though he is in the ITI). 
        % If you dont want to allow it, add the following code to input_to_statechange below. (but it will mean that the trial code will be aborted and will be again sent). centerLick,'start_rotary_scope', correctLick,'start_rotary_scope', errorLick,'start_rotary_scope'
        
        % trialCode1 :  Right here, before the start-tone, is when the trial code gets sent.
        
%         durCode = 0;
        stateNum = 0;
        for pair = 1:size(code, 2)
            stateNum = stateNum + 1;
            stateName = ['trialCode' num2str(stateNum)];
            nextState = [stateName 'Low'];
            
            % High state (send bars, ie send scopeTrial pulse). Always
            % followed by low state.
            sma = add_state(sma, 'name', stateName, ...
                'self_timer', codeModuleDurs(code(1, pair)), ...
                'output_actions', {'DOut', trial_ports + scopeTrial}, ...
                'input_to_statechange', {'Tup', nextState}); % right now you are allowing the mouse to lick during trialCode time (even though he is in the ITI). If you dont want to allow it, add the following, but it will mean that the trial code will be aborted and will be again sent. centerLick,'start_rotary_scope', correctLick,'start_rotary_scope', errorLick,'start_rotary_scope'
            
            
            % Low state (send spaces, ie dont send scopeTrial pulse).
            % Either followed by high state or (when the code is all sent) by
            % state wait_for_initiation.
            stateName = nextState;
            if pair == size(code, 2)
                nextState = 'wait_for_initiation';
            else
                nextState = ['trialCode' num2str(stateNum + 1)];
            end
            sma = add_state(sma, 'name', stateName, ...
                'self_timer', codeModuleDurs(code(2, pair)), ...
                'output_actions', {'DOut', trial_ports}, ...
                'input_to_statechange', {'Tup', nextState}); % right now you are allowing the mouse to lick during trialCode time (even though he is in the ITI). If you dont want to allow it, add the following, but it will mean that the trial code will be aborted and will be again sent. centerLick,'start_rotary_scope', correctLick,'start_rotary_scope', errorLick,'start_rotary_scope'
            
%             durCode = durCode + sum(codeModuleDurs(code(:, pair)));
        end
        

        %% Playing the start tone and waiting for the mouse to initiate the trial.

        % wait_for_initiation
        sma = add_state(sma, 'name', 'wait_for_initiation',...
            'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
            'output_actions', {'DOut', trial_ports, 'SoundOut', startTone},...
            'input_to_statechange', {'Tup','wait_for_initiation2', ... %used to be iti but i want to keep record of it.
            centerLick,'stim_delay', correctLick,'wrong_initiation', errorLick,'wrong_initiation'}); % wrong_initiation simply leads to state iti.
        
        % wait_for_initiation2
        sma = add_state(sma, 'name', 'wait_for_initiation2',...
            'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
            'output_actions', {'DOut', trial_ports, 'SoundOut', startTone},...
            'input_to_statechange', {'Tup','wait_for_initiation', ... %used to be iti but i want to keep record of it.
            centerLick,'stim_delay', correctLick,'wrong_initiation', errorLick,'wrong_initiation'}); % wrong_initiation simply leads to state iti.
        
            
%{            
        % wait_for_initiation        
        if ~value(CenterPoke_amount)
            % wait_for_initiation
            sma = add_state(sma, 'name', 'wait_for_initiation',...
                'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
                'output_actions', {'DOut', trial_ports, 'SoundOut', startTone, 'SchedWaveTrig', 'center_unconst'},...
                'input_to_statechange', {'Tup','wait_for_initiation2', ... %used to be iti but i want to keep record of it.
                centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation'}); % wrong_initiation simply leads to state iti.
            
            % wait_for_initiation2
            sma = add_state(sma, 'name', 'wait_for_initiation2',...
                'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
                'output_actions', {'DOut', trial_ports, 'SoundOut', startTone},...
                'input_to_statechange', {'Tup','wait_for_initiation', ... %used to be iti but i want to keep record of it.
                centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation'}); % wrong_initiation simply leads to state iti.
            
        else % give a large drop of water in the center if the mouse did not initiate a trial in [180 300] seconds. water amount = water_duration_ave*value(CenterPoke_amount)
            
            % wait_for_initiation
            sma = add_state(sma, 'name', 'wait_for_initiation',...
                'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
                'output_actions', {'DOut', trial_ports, 'SoundOut', startTone, 'SchedWaveTrig', 'center_unconst'},...
                'input_to_statechange', {'Tup','wait_for_initiation2', ... %used to be iti but i want to keep record of it.
                centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation',...
                'center_unconst_In','give_center_unconst'}); % wrong_initiation simply leads to state iti.
            
            % wait_for_initiation2
            sma = add_state(sma, 'name', 'wait_for_initiation2',...
                'self_timer', value(TimeToInitiate), ... % 60 sec value(TimeToInitiate)
                'output_actions', {'DOut', trial_ports, 'SoundOut', startTone},...
                'input_to_statechange', {'Tup','wait_for_initiation', ... %used to be iti but i want to keep record of it.
                centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation',...
                'center_unconst_In','give_center_unconst'}); % wrong_initiation simply leads to state iti.

        end        
        
        % give_center_unconst: % give a large drop of water in the center if the mouse did not initiate a trial in [180 300] seconds. water amount = water_duration_ave*value(CenterPoke_amount)
        sma = add_state(sma, 'name', 'give_center_unconst',...
            'self_timer', water_duration_ave*value(CenterPoke_amount), ...
            'output_actions', {'DOut', trial_ports + cwater}, ...
            'input_to_statechange', {'Tup','wait_for_initiation', ... %used to be iti but i want to keep record of it.
            centerLick,'stop_center_unconst', correctLick,'wrong_initiation', errorLick,'wrong_initiation'});              
        
        
        
        % stop_center_unconst (stop the center_unconst scheduled wave)
        sma = add_state(sma, 'name', 'stop_center_unconst',...
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SchedWaveTrig', '-center_unconst'},...
            'input_to_statechange', {'Tup','stim_delay'});
%}            
        
        %% Mouse initiating the trial and stimulus playing.
        % stim_delay % 
        sma = add_state(sma, 'name', 'stim_delay',...
            'self_timer', value(PreStimDelay),...
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','wait_stim', correctLick,'early_decision', errorLick,'early_decision'}); % early_decision is just like iti. Defined to keep trach of early decisions.
        
        % wait_stim; in this state stim will start playing; its duration is
        % waitDur + extraStimDur + stimDur_aftRew; if stimDur_diff is
        % non-zero, then stim duration will be simDur_diff +
        % extraStimDur + stimDur_aftRew
        sma = add_state(sma, 'name', 'wait_stim',...
            'self_timer', wait_dur + value(PostStimDelay),...
            'output_actions', {'DOut', trial_ports, 'SoundOut', flickStim},...
            'input_to_statechange', {'Tup','lickcenter_again', correctLick,'early_decision0', errorLick,'early_decision0'}); %randomState


        % the following 2 states were added in fn_lick_1_3
        % lickcenter_again
        sma = add_state(sma, 'name', 'lickcenter_again',...
            'self_timer', value(CenterAgain), ... % .3 you may want to make this shorter as the training gets better.
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {centerLick, 'center_reward', correctLick,'early_decision0', errorLick,'early_decision0', 'Tup','did_not_lickagain'});

        % center_reward. 
        % randomSate is direct_water for Direct, and wait_for_decision for Allow correction and Choose side.
        sma = add_state(sma, 'name', 'center_reward', ...
            'self_timer', value(CenterWater),...
            'output_actions', {'DOut', cwater + trial_ports, 'SoundOut',goTone},...
            'input_to_statechange', {'Tup', randomState});
        
        
        %% Mouse choosing a side spout.
        % states below come after the mouse correctly initiates a trial (ie he does 1st lick and commit lick and receives center reward). 
 
        % This state will come after center_reward if in Direct stage (so randomState is direct_water).
        sma  = add_state(sma, 'name', 'direct_water',...
                'self_timer', water_duration,...
                'output_actions', {'DOut', water_delivery + trial_ports},...
                'input_to_statechange', {'Tup', 'wait_for_decision'});       
            
            
        % This state will come after direct_water (for Direct stage).
        if strcmp(value(rewardStage), 'Direct')
            sma = add_state(sma, 'name', 'wait_for_decision',...
            'self_timer', value(TimeToChoose)+3000,...
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','did_not_choose', correctLick,'direct_correct'});            
        
        
        
        % This state will come after center_reward if in Allow correction and Choose side (so random state is wait_for_decision).
        else 
            % wait_for_decision. Define correct and incorrect choice as licking for .2sec
            sma = add_state(sma, 'name', 'wait_for_decision',...
                'self_timer', value(TimeToChoose),... +3000
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','did_not_choose', correctLick,'correctlick_again_wait', errorLick,'errorlick_again_wait'});
            
            % correctlick_again_wait. mouse will not receive reward unless he licks again on the correct side after .2
            sma = add_state(sma, 'name', 'correctlick_again_wait',...
                'self_timer', sidelick_dur,... % perhaps gradually increase this value to make the choices more costly and less impulsive.
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','correctlick_again', errorLick,'errorlick_again_wait'});
                           

            % errorlick_again_wait. if he errorlicks, he will go to a again_wait state (.2), if he licks again after that, now it is actually an errorlick. so state punish_allowcorrect will happen.
            sma = add_state(sma, 'name', 'errorlick_again_wait',...
                'self_timer', sidelick_dur,...
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','errorlick_again', correctLick,'correctlick_again_wait'});
            
            
            % correctlick_again. he has a window of .2 to lick again to receive reward.  perhaps later change Tup to 'did_not_sidelickagain', or define a schedulewave. (fn attention) % you changed to did_not_choose on 3/19/15. it used to be 'wait_for_decision'
            sma = add_state(sma, 'name', 'correctlick_again',...
                'self_timer', value(SideAgain),... % .4; .3
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','wait_for_decision2', correctLick, 'reward', errorLick,'errorlick_again_wait'});
                        
            
            % errorlick_again. he has a window of .2 to lick again, if he does punishment happens. perhaps later change Tup to 'did_not_choose'
            if strcmp(value(rewardStage), 'Allow correction')
                
                sma = add_state(sma, 'name', 'errorlick_again',...
                    'self_timer', value(SideAgain),...
                    'output_actions', {'DOut', trial_ports},...
                    'input_to_statechange', {'Tup','wait_for_decision2', errorLick, 'punish_allowcorrection', correctLick,'correctlick_again_wait'});
            
            elseif strcmp(value(rewardStage), 'Choose side')
                
                sma = add_state(sma, 'name', 'errorlick_again',...
                    'self_timer', value(SideAgain),...
                    'output_actions', {'DOut', trial_ports},...
                    'input_to_statechange', {'Tup','wait_for_decision2', errorLick, 'punish', correctLick,'correctlick_again_wait'});
            end
            
            
            % wait_for_decision2 : if the mouse didn't lickAgain, give him a second chance to respond properly (ie licking again after sideLickDur).
            sma = add_state(sma, 'name', 'wait_for_decision2',...
                'self_timer', value(TimeToChoose2),... % 4
                'output_actions', {'DOut', trial_ports},...
                'input_to_statechange', {'Tup','did_not_sidelickagain', correctLick,'correctlick_again_wait', errorLick,'errorlick_again_wait'});           
            
        end
        
        %{
        elseif strcmp(value(rewardStage), 'Allow correction')
%             sma = add_state(sma, 'name', 'wait_for_decision',...
%                 'self_timer', value(TimeToChoose),... +3000
%                 'input_to_statechange', {'Tup','did_not_choose', correctLick,'reward'}); % did_not_choose is same as stopSound, but I am defining it so I can have a record of did_not_chooses.
            
            sma = add_state(sma, 'name', 'wait_for_decision',...
                'self_timer', value(TimeToChoose),... +3000
                'input_to_statechange', {'Tup','did_not_choose', correctLick,'reward', errorLick,'punish_allowcorrection'}); % did_not_choose is same as stopSound, but I am defining it so I can have a record of did_not_chooses.
            
            % punishment if allow correction
            sma = add_state(sma, 'name', 'punish_allowcorrection',...
                'self_timer', 1E-6,...
                'output_actions', {'SoundOut', incorrectTone},...            
                'input_to_statechange', {'Tup','wait_for_decision', correctLick,'reward'});
            
            
        elseif strcmp(value(rewardStage), 'Choose side')
            sma = add_state(sma, 'name', 'wait_for_decision',...
                'self_timer', value(TimeToChoose),...
                'input_to_statechange', {'Tup','did_not_choose', correctLick,'reward', errorLick,'punish'});
        end
        %}
        
        
        %% states defining the outcome of a trial.
        
        % wrong_initiation
        sma = add_state(sma, 'name', 'wrong_initiation',...
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SchedWaveTrig', '-center_unconst'},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        % early_decision0
        sma = add_state(sma, 'name', 'early_decision0',... % like early_decision, but also stops the stimulus. 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','early_decision'});        
        
        % early_decision
        sma = add_state(sma, 'name', 'early_decision',... 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        
        % did_not_lickagain
        sma = add_state(sma, 'name', 'did_not_lickagain',... % like early_decision, but also stops the stimulus. 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        
        % did_not_choose
        sma = add_state(sma, 'name', 'did_not_choose',... 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});

        
        
        % did_not_sidelickagain
        sma = add_state(sma, 'name', 'did_not_sidelickagain',... 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        % direct_correct (when the animal correctLicks in the Direct stage).  just like did_not_choose.
        sma = add_state(sma, 'name', 'direct_correct',...
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        
        
        
        
        % reward
%         sma = add_state(sma, 'name', 'reward',...
%             'self_timer', water_duration,...
%             'output_actions', {'SoundOut', -flickStim, 'DOut', water_delivery},...
%             'input_to_statechange', {'Tup','iti'});
        
        % reward: remember in this new ways, ITI if computed from the time of reward to the start of the next trial will be 1 sec longer that value(ITI). also remember that ITI does not include the duration of check_next_trial_ready, which is about 1.5 sec for my protocol.
        sma = add_state(sma, 'name', 'reward',...
            'self_timer', water_duration,...
            'output_actions', {'DOut', water_delivery + trial_ports, 'SoundOut', -incorrectTone},...
            'input_to_statechange', {'Tup','stopstim_pre'});
        
        % keep playing the stim for StimDur_aftRew sec. after that or if the animal errorLicks the stim stops.
        % but remember if the stim has already stopped, then this state
        % wont mean anything! For this reason it makes no sense to make this state last for stimdur_aftrew... fix this for your new mice!
        sma = add_state(sma, 'name', 'stopstim_pre',...
            'self_timer', stimdur_aftrew,... % 1
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','reward_stopstim', errorLick,'reward_stopstim'});
        
        sma = add_state(sma, 'name', 'reward_stopstim',...
            'self_timer', 1E-6,...
            'output_actions',  {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});        
        
        
        

        % punish_allowcorrection (consider adding time out too)
        sma = add_state(sma, 'name', 'punish_allowcorrection',...
            'self_timer', value(TimeToChoose2),... % he has timeToChoose sec to correct lick.
            'output_actions', {'DOut', trial_ports, 'SoundOut', incorrectTone},...            
            'input_to_statechange', {'Tup','punish_allowcorrection_done', correctLick,'correctlick_again_wait'});
        
        sma = add_state(sma, 'name', 'punish_allowcorrection_done',... 
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
        

        % punish (if choose side)
        sma = add_state(sma, 'name', 'punish',...
            'self_timer', 1E-6,...
            'output_actions', {'DOut', trial_ports, 'SoundOut', -flickStim},...
            'input_to_statechange', {'Tup','punish_timeout'});

        % Auditory feedback and Time out punish 
        sma = add_state(sma, 'name', 'punish_timeout',...
            'self_timer', value(errorTimeout),...
            'output_actions', {'DOut', trial_ports, 'SoundOut', incorrectTone},...
            'input_to_statechange', {'Tup','stop_rotary_scope'});
                
         
        
        %% end of states. (another 500ms of the ITI)
        %%% stop_rotary_scope --> last state that includes scopeTTL and
        %%% ardttl. you set it to 500ms, bc you want to record
        %%% imaging and rotary data during ITI.
        sma = add_state(sma, 'name', 'stop_rotary_scope',...
            'self_timer', value(stopScopeDur),...
            'output_actions', {'DOut', trial_ports},...
            'input_to_statechange', {'Tup','check_next_trial_ready'});
        
        
%         % last_state
%         sma = add_state(sma, 'name', 'last_state',...
%             'self_timer', 1E-6,...
%             'input_to_statechange', {'Tup','check_next_trial_ready'});
        
        
        % parsedEvents.states.ending_state will indicate one of the states below which were sent to the assembler.
%         dispatcher('send_assembler', sma, {'punish_timeout', 'punish_allowcorrection_done', 'reward_stopstim', 'direct_correct', 'did_not_choose', 'did_not_sidelickagain', 'did_not_lickagain', 'early_decision', 'wrong_initiation'});
        dispatcher('send_assembler', sma, {'stop_rotary_scope'});
%         dispatcher('send_assembler', sma, {'last_state'});
        



        %---------------------------------------------------------------
        %          CASE TRIAL_COMPLETED
        %---------------------------------------------------------------
    case 'trial_completed'
        
        PokesPlotSection(obj, 'trial_completed');
        % PerformancePlotSection(obj, 'update', n_done_trials, value(HitHistory), value(SidesListBool) == 0);
        
        % saving data
        if n_done_trials > 1
            %stimulus_strengths = value(StimulusStrengths);
            %wait_duration_trials = value(WaitDurationTrials);
            %visual_audio_trials = value(VisualAudioTrials);
            %inter_stim_intervals = value(InterStimIntervals);
            all_data = value(AllData);
            save(value(FileName), 'parsed_events_history', 'all_data');
            % save(value(FileName), 'parsed_events_history', 'stimulus_strengths', 'wait_duration_trials', 'visual_audio_trials', 'inter_stim_intervals');
        end
        
        %---------------------------------------------------------------
        %          CASE UPDATE
        %---------------------------------------------------------------
    case 'update'
        PokesPlotSection(obj, 'update');
        % my_parsed_events.states
        
        %---------------------------------------------------------------
        %          CASE CLOSE
        %---------------------------------------------------------------
    case 'close'
        
        %{
      if value(RecordingSession) == 1
          NlxDisconnectFromServer;
      end
        %}
        
        PokesPlotSection(obj, 'close');
        if exist('mygui', 'var') && isa(mygui, 'SoloParamHandle') && ishandle(value(mygui)),
            delete(value(mygui));
        end;
        
    case 'load_settings_recent'
        load_solouiparamvalues(value(SubjectName), 'interactive', 0);
        
    case 'load_settings_file'
        load_solouiparamvalues(value(SubjectName), 'interactive', 1);
        
    case 'save_settings'
        save_solouiparamvalues(value(SubjectName), 'interactive', 0);
        
    otherwise,
        warning('Unknown action! "%s"\n', action);
end;

return;
end





%% Definition of functions used in this script
function rand_exp = generate_random_exp (lambda, minimum, maximum)

%     x = nan;
%     while isnan(x) || x < minimum || x > maximum
%        x = log(1-rand)/(-lambda);
%     end
%     random_delay = x;

x = -log(rand)/lambda;
rand_exp = mod(x, maximum - minimum) + minimum; % add minimum value instead of using as threshold % Matt's MOD solution for this is very elegant and works great.

end



function [newModeRightArray, newSuccessArray] = ...
    updateAntiBiasArrays(modeRightArray, successArray, visOrAud, outcome, ...
    prevCorrectSide, prevLR, prevSuccess, antiBiasTau, wentRight)
% Based on what happened on the last completed trial, update our beliefs
% about the animal's biases. modeRightArray tracks how likely he is to go
% right for each modality. successArray tracks how likely he is to succeed
% for left or right given what he did on the previous trial.
%
% Note: we'll actually use antiBiasTau * 3 for updating the
% modality-related side bias. If we don't, and there's only one modality,
% the updates will cause oscillation against a perfectly consistent
% strategy.

% For an exponential function, the pdf = (1/tau)*e^(-t/tau)
% The integral from 0 to 1 is [1 - e^(-1/tau)]
% This lets us do exponential decay using only the current
% outcome and previous biases
antiAlternationW = 1 - exp(-1/(3*antiBiasTau));
antiBiasW = 1 - exp(-1/antiBiasTau);

% modeRightArray -- how often he's gone right for each modality
modality = 2 + visOrAud;
newModeRightArray = modeRightArray;
if ~isnan(wentRight)
    newModeRightArray(modality) = antiAlternationW * wentRight + (1 - antiAlternationW) * modeRightArray(modality);
end

% Can only update arrays if we already had a trial in the history (since we
% have a two-trial dependence)
newSuccessArray = successArray;
if ~isnan(prevLR)
    newSuccessArray(prevLR, prevSuccess + 1, prevCorrectSide) = antiBiasW * (outcome > 0) + ...
        (1-antiBiasW) * successArray(prevLR, prevSuccess + 1, prevCorrectSide);
end

end


function pLeft = getAntiBiasPLeft(successArray, modeRightArray, modality, ...
    antiBiasStrength, prevLR, prevSuccess)

% Find the relevant part of the SuccessArray and ModeRightArray
successPair = squeeze(successArray(prevLR, prevSuccess + 1, :));

modeRight = modeRightArray(modality);


% Based on the previous successes on this type of trial,
% preferentially choose the harder option

succSum = sum(successPair);

pLM = modeRight;  % prob desired for left based on modality-specific bias
pLT = successPair(2) / succSum;  % same based on prev trial
iVar2M = 1 / (pLM - 1/2) ^ 2; % inverse variance for modality
iVar2T = 1 / (pLT - 1/2) ^ 2; % inverse variance for trial history

if succSum == 0 || iVar2T > 10000
    % Handle degenerate cases, trial history uninformative
    pLeft = pLM;
elseif iVar2M > 10000
    % Handle degenerate cases, modality bias uninformative
    pLeft = pLT;
else
    % The interesting case... combine optimally
    pLeft = pLM * (iVar2T / (iVar2M + iVar2T)) + pLT * iVar2M / (iVar2M + iVar2T);
end

% Weight pLeft from anti-bias by antiBiasStrength
pLeft = antiBiasStrength * pLeft + (1 - antiBiasStrength) * 0.5;

end



function coin_th = trSeq(coin, th_consecutive)
% Generate a sequence of trials without more than th_consectuive trials of the same type
% in a row (Farzaneh)

% identify consecutive 0s > th_consectuve
evd = coin;
%         evd(end+1) = 1;
evd = [1 evd 1];
f1 = find(evd==1)-1; % you subtract 1 bc you added an element to the beginning of evd.
evdist = diff(f1)-1;

f11 = f1(find(evdist > th_consecutive )); 
f12 = f1(find(evdist > th_consecutive)+1);
f_ldist = [f11;f12]';


% identify consecutive 1s > th_consecutive
evd = coin;
evd(coin==0) = 1;
evd(coin==1) = 0;

evd = [1 evd 1];
f1 = find(evd==1)-1;
evdist = diff(f1)-1;

f11 = f1(find(evdist > th_consecutive )); 
f12 = f1(find(evdist > th_consecutive)+1);
f_ldist1 = [f11;f12]';


% remove consecutive 0s and 1s
for fi = 1:size(f_ldist,1)
    coin(f_ldist(fi,1)+1 : f_ldist(fi,2)) = NaN;        
end

for fi = 1:size(f_ldist1,1)
    coin(f_ldist1(fi,1)+1 : f_ldist1(fi,2)) = NaN;        
end

coin(isnan(coin)) = [];
coin_th = coin;

end



function [evnum, hiSall, loSall, min_evnum, max_evnum, categ_bound_num] = set_evnum(stimdur, evdur, gapdur, max_iei_accepted, categ_bound_Hz, vis_strength_prop)
% Set number of events (evnum) for the current stimulus for both the
% low-rate (evnum(1)) and high-rate (evnum(2)) cases. (Farzaneh)

% min and max number of events for the given stimulus duration.
max_evnum = floor(stimdur/(evdur+gapdur));
if stimdur >= 890 && stimdur <= 1100
    min_evnum = round(stimdur/(max_iei_accepted+evdur))+2; % it will take very long for 5 events, so use 6 as the minimum.
elseif stimdur > 250
    min_evnum = round(stimdur/(max_iei_accepted+evdur))+1; % this formula works well.
else 
    min_evnum = 1;
end

categ_bound_num = categ_bound_Hz*stimdur/1000; % category bondary (number of events) for the current stimulus.

% generate a vector in order to group number of events into hard, medium-hard, medium-easy, easy
if (max_evnum - floor(categ_bound_num))>=4 % this value will be <4 when waitdur is very short (eg the beginning of training). In this case don't set vec_allStrength, and just use max_evnum as evnum.
%     vec_allStrength = floor(categ_bound_num) : ceil((max_evnum - floor(categ_bound_num))/4) : max_evnum; % devide the range (from categ boundary to the max evnum) into 4 groups.
%     vec_allStrength = [vec_allStrength, max_evnum];
    denom = floor((max_evnum - floor(categ_bound_num)) / 4);
    remin = mod(max_evnum - floor(categ_bound_num), 4);

    if remin==0 % the code for remin<3 works for this case too, but you are writing it bc the code is much simpler if remin=0;.
        vec_allStrength = [floor(categ_bound_num)+1: denom: max_evnum+1];
    elseif remin<3 % add all the extra values (=remin, either 1 or 2 values) to the hard condition. % by value I mean number of events in your stimulus.
        v1 = floor(categ_bound_num)+1: floor(categ_bound_num)+remin+denom;
        vrest = floor(categ_bound_num)+remin+denom+1: denom:  max_evnum;
        vec_allStrength = [v1(1), vrest, max_evnum+1];
    else  % since you are dividing by 4, this case only happens when remin=3. If so, add 2 of the extra values to the hard condition and 1 of them to the med-hard condition.
        v1 = floor(categ_bound_num)+1: floor(categ_bound_num)+remin-1+denom;
        v2 = floor(categ_bound_num)+remin-1+denom+1: floor(categ_bound_num)+remin-1+denom+1+denom;
        vrest = floor(categ_bound_num)+remin-1+denom+1+denom+1: denom:  max_evnum+1;
        vec_allStrength = [v1(1), v2(1), vrest, max_evnum+1];
    end

    if length(vec_allStrength)~=5 
        if (length(vec_allStrength)~=6 || vec_allStrength(end-1)~= vec_allStrength(end))
            error('something is wrong with how you are defining the easy-hard categories!!')
        end
    end

    % pick a random strength group (hard, medium-hard, medium-easy, easy) from vis_strength_prop 
    % (probability that a particular group happens is given by vis_strength_prop)
    v = [0 cumsum(vis_strength_prop)];
    v(end) = v(end)+.01;
    r = rand;
    [n,i] = histc(r,v);

%     hiS = vec_allStrength(i)+1:vec_allStrength(i+1); % high rates
    hiS = vec_allStrength(i):vec_allStrength(i+1)-1;

else % this happens at very short wait durations. 
    hiS = max_evnum;
end
% hiS = max_evnum;
loS = min_evnum + max_evnum - hiS; % low rates
loS(loS<min_evnum) = [];

% set event numbers for all 4 difficulty levels.
hiSall = cell(1,4); hiSall(:) = {NaN};
loSall = cell(1,4); loSall(:) = {NaN};
if (max_evnum - floor(categ_bound_num))>=4
    for i = 1:4
        hiSall{i} = vec_allStrength(i) : vec_allStrength(i+1) - 1;
        loSall{i} = min_evnum + max_evnum - hiSall{i};
    end
else
    hiSall{1} = hiS;
    loSall{1} = loS;
end


% pick 2 random values of evnum (number of events) from loS and hiS (probability
% that a particular evnum happens is equal for all evnums in strengthGroup)
vthisS = 0: 1/length(loS) :1;
r = 1; while any(r==1), r = rand; end
[n,i] = histc(r,vthisS);
evnum(1) = loS(i);

vthisS = 0: 1/length(hiS) :1;
r = 1; while any(r==1), r = rand; end
[n,i] = histc(r,vthisS);
evnum(2) = hiS(i);

%{
strengthGroup = [loS, hiS]; 
vthisS = 0: 1/length(strengthGroup) :1;
r = 1; while r==1, r = rand; end
[~,i] = histc(r,vthisS);
evnum = strengthGroup(i);
%}

end



function ieis = set_stim_fn(stimdur, evdur, gapdur, evnum, max_iei_accepted) % inputs must be in ms. output is a set of intervals (in ms)
% generate a poisson stimulus (Farzaneh) % ieis: inter-event intervals.

max_evnum = floor(stimdur/(evdur+gapdur));
if evnum > max_evnum
    disp('stimulus duration (stimdur) is too short for this number of events (evnum). max_evnum is used instead.')
    evnum = max_evnum;
end


if stimdur >= 890 && stimdur <= 1100
    min_evnum = round(stimdur/(max_iei_accepted+evdur))+2; % it will take very long for 5 events, so use 6 as the minimum.
elseif stimdur > 250
    min_evnum = round(stimdur/(max_iei_accepted+evdur))+1; % this formula works well.
else 
    min_evnum = 1;
end

if evnum < min_evnum
    disp('number of events entered is too small. min_evnum is used instead.')
    evnum = min_evnum;
end

evdur_temp = 1; % event duration in ms
visdur_temp = stimdur - evnum*(evdur+gapdur-evdur_temp);

lambda = evnum/visdur_temp;


vis_stim = 0;
while sum(diff([0 vis_stim 0])==-1)~=evnum
    % generate a poisson sequence.
    evorno = poissrnd(lambda, 1, stimdur*50);
    evorno((evorno>1)) = 1; % in case there are some values bigger than 1, turn them to 1
    evorno = evorno(find(evorno,1):end); % we want to have an event at the beginning of the stimulus.
    % figure; plot(evorno)

    % make sure there are no events in the poisson process with
    % iei>max_iei_accepted. if there are any, remove those events(and
    % their ieis.)
    evd = evorno;
    evd(end+1) = 1;
    f1 = find(evd==1);
    evdist = diff(f1)-1;

    f11 = f1(find(evdist > max_iei_accepted-gapdur )); % remember each event (element=1) in the poisson (evorno) represents 1event + 1gap in the vis_stim.
    f12 = f1(find(evdist > max_iei_accepted-gapdur)+1);
    f_ldist = [f11;f12]';
    for fi = 1:size(f_ldist,1)
        evorno(f_ldist(fi,1)+1:f_ldist(fi,2)) = NaN;        
    end
    evorno(isnan(evorno)) = [];
    %

    % set your vis_stim using the poisson sequence.
    vis_stim = zeros(1,stimdur); 
    i_poiss = 1; 
    i_vstim = 1;
%         i_poiss1 = 1;
    while i_vstim <= length(vis_stim)-(evdur-1)
        %{
        % here you make sure there are no ieis>max_iei_accepted. but this method is not good bc it results in an increase in the number of ieis at 200ms. 
        if i_poiss > i_poiss1+(max_iei_accepted-gapdur)+1
            i_poiss = i_poiss + find(evorno(i_poiss:i_poiss+500)==1, 1)-1;
        end
        %}
        if evorno(i_poiss)==1 % make sure the event lasts for evdur, and followed by a gap of duration gapdur. 
%                 i_poiss1 = i_poiss;
            vis_stim(i_vstim : i_vstim+(evdur-1)) = 1;
            vis_stim(i_vstim+evdur : i_vstim+(evdur+gapdur-1)) = 0;
            i_vstim = i_vstim+(evdur+gapdur);
        else
            i_vstim = i_vstim+1;
        end
        i_poiss = i_poiss+1; % go through the poisson sequence you generated.
%             [i_poiss1, i_poiss, i_vstim]
%             pause
    end
end
vis_stim(stimdur+1:end) = [];

%     plot(vis_stim), % pause
%     numev_all(iit) = sum(diff([0 vis_stim 0])==-1);

%%% find inter-event intervals
a = [1 diff(vis_stim) 1];
ieis = diff(find(a==1))-evdur;

end



function signal_visual = create_visual_signal_fn(ieis_sec, srate, brightness, flashdur, offset)

timevec = (0 : 1/srate : flashdur);
freq = 200*brightness; % we can't detect a 300 Hz flicker %FN: the freq must be high enough such that flicker fusion happens. The higher the frequency, the brighter the LED signal will be.
sine_wave = sin(2*pi * freq * timevec);

[pks,locs] = findpeaks(sine_wave);

flash = zeros(size(sine_wave));
flash(locs) = 1;
offset_dark = zeros(1, round(srate * offset));

% signal_visual = [offset_dark flash];
signal_visual = offset_dark;

for i = 1:length(ieis_sec)
    dark = zeros(1, round(srate * ieis_sec(i)));
%     signal_visual = [signal_visual dark flash];
    signal_visual = [signal_visual flash dark]; % Farzaneh: not sure why
%     to do it this way which adds an extra event at the end, hence changes
%     the stimulus rate!! sounds like a bad idea!
end

end



function signal_audio = create_audio_signal_fn(ieis_sec, srate, tone_duration, ...
    sound_carrier, sound_loudness, offset, noise_loudness, CalibrationModelParams)

% play sound using this rate
% srate = 44100;
freq = 15000; % Hz
timevec = (0 : 1/srate : tone_duration);
% a*x + b = y; x=SPL; y = 10*log10(toneAmp); a  & b are the coefficients of the polyfit done at the begining of this script.
signal_amplitude = 10^(1/10*(CalibrationModelParams(1)*sound_loudness + CalibrationModelParams(2)));
noise_amplitude = 10^(1/10*(CalibrationModelParams(1)*noise_loudness + CalibrationModelParams(2)));

envelope = sin([0:pi/(srate*tone_duration):pi]); % FN: you want envelope to have the same length as the sound,  and only  do half a phase, i.e. go from sin(0) to sin(pi).

if strcmp(sound_carrier, '15 KHz + envelope')
    soundpart = signal_amplitude * (sin(2*pi * freq * timevec) .* envelope);
elseif strcmp(sound_carrier, '15 KHz')
    soundpart = signal_amplitude * (sin(2*pi * freq * timevec));
else
    wnoise = signal_amplitude * 2 * (rand(1,length(timevec))-0.5); % FN: here they are making sure that wnoise is within [-signal_amp signal_amp].i.e. a+(b-a)*rand
    soundpart = wnoise .* envelope; % FN: David doesn't use envelope for white noise, but I think it is a good idea to make white noise like the pure tone in terms of the amplitude.
end

% figure; plot(soundpart);
offset_part = zeros(1, round(srate * offset));

% line below adds an event at t = 0 + offset
% signal_audio = [offset_part soundpart];
signal_audio = offset_part;

for i = 1:length(ieis_sec)
    dark = zeros(1, round(srate * ieis_sec(i)));
%     signal_audio = [signal_audio dark soundpart];
    signal_audio = [signal_audio soundpart dark]; % Farzaneh: not sure why
%     to do it this way which adds an extra event at the end, hence changes
%     the stimulus rate!! sounds like a bad idea!
end

end
    

