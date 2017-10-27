function BpodImager_delayRegressModel(cPath,Animal,Rec)

if ~strcmpi(cPath(end),filesep)
    cPath = [cPath filesep];
end

%general variables
Paradigm = 'SpatialDisc';
cPath = [cPath Animal filesep Paradigm filesep Rec filesep]; %Widefield data path
sPath = ['U:\space_managed_data\BpodImager\Animals\' Animal filesep Paradigm filesep Rec filesep]; %server data path. not used on hpc.
sRate = 40; % Sampling rate of imaging in Hz

% trial structure variables
% preStimDur = 3;                 % Duration of trial before stimulus onset in seconds
% postStimDur = 115 / sRate;      % Duration of trial after stimulus onset in seconds
% trialSegments = [2 1 1.7 1 1];  % Duration of  different segments in each trial in seconds. This refers to baseline, lever grab, stimulus, delay and decision phase and usually there are 1s each.

preStimDur = 2;                 % Duration of trial before stimulus onset in seconds
postStimDur = 2;      % Duration of trial after stimulus onset in seconds
trialSegments = [1 1 1 1];  % Duration of  different segments in each trial in seconds. This refers to baseline, lever grab, stimulus, delay and decision phase and usually there are 1s each.

% ridge regression variables
ridgeFolds = 2;     %number of folds for cross validation
ridgeVal = 500;     %initial estimate for ridge parameter

%other variables
mPreTime = 0.5;     % precede motor events to capture preparatory activity in seconds
mPostTime = 1;      % follow motor events for mPostStim in seconds
tapDur = 0.25;      % minimum time of lever contact, required to count as a proper grab.
smth = 5;           % filter length when smoothing video data
smth = smth-1+mod(smth,2); % ensure kernel length is odd
piezoLine = 2;      % channel in the analog data that contains data from piezo sensor
stimLine = 6;      % channel in the analog data that contains stimulus trigger.

bhvDimCnt1 = 500;    % number of dimensions from behavioral videos that are used as regressors.
bhvDimCnt2 = bhvDimCnt1 / 10;    % number of dimensions from behavioral videos that are used after 2nd SVD where videos are merged.
bhvOffset = 5;     % offset when creating shifter regressors from bhv video data. Will create two sets of additional regressors that are shifted by bhvOffset frames in either direction. Sampling rate is usually 30Hz.
newRun = true;     %flag to indicate that all values should be recomputed (including ridge regression and correlation maps)

%% load behavior data
if exist([cPath 'BehaviorVideo' filesep 'SVD_Cam 1.mat'],'file') ~= 2 %check if svd behavior exists on hdd and pull from server otherwise
    if exist([cPath 'BehaviorVideo' filesep]) ~= 2
        mkdir([cPath 'BehaviorVideo' filesep]);
    end
    copyfile([sPath 'BehaviorVideo' filesep 'SVD_Cam 1.mat'],[cPath 'BehaviorVideo' filesep 'SVD_Cam 1.mat']);
    copyfile([sPath 'BehaviorVideo' filesep 'SVD_Cam 2.mat'],[cPath 'BehaviorVideo' filesep 'SVD_Cam 2.mat']);
end

load([cPath 'BehaviorVideo' filesep 'SVD_Cam 1.mat']);
V1 = V';
U1 = U;
frameTimes1 = totalFrameTimes;

load([cPath 'BehaviorVideo' filesep 'SVD_Cam 2.mat']);
V2 = V';
U2 = U;
frameTimes2 = totalFrameTimes;

%% load data
load([cPath 'mask.mat'])
load([cPath 'Vc.mat'])

bhvFile = dir([cPath filesep Animal '_' Paradigm '*.mat']);
load([cPath bhvFile(1).name]); %load behavior data

% ensure there are not too many trials in Vc
ind = trials > SessionData.nTrials;
trials(ind) = [];
Vc(:,:,ind) = [];
[dims, frames, trialCnt] = size(Vc);

bhv = selectBehaviorTrials(SessionData,trials); %only use completed trials that are in the Vc dataset
bhv.stimOn = ones(1,length(bhv.Rewarded)) * 0.2;

%% find events in BPod time.
% All timestamps are relative to stimulus onset event to synchronize to imaging data later

% pre-allocate vectors
lickL = cell(1,trialCnt);
lickR = cell(1,trialCnt);
leverIn = NaN(1,trialCnt);
levGrabL = cell(1,trialCnt);
levGrabR = cell(1,trialCnt);
levReleaseL = cell(1,trialCnt);
levReleaseR = cell(1,trialCnt);
water = NaN(1,trialCnt);

for iTrials = 1:trialCnt
    
    try
        stimTime(iTrials) = bhv.RawEvents.Trial{iTrials}.Events.Wire3High; %time of stimulus onset - measured from soundcard
    catch
        stimTime(iTrials) = NaN;
    end
    
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Port1In') %check for licks
        lickL{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Port1In;
        lickL{iTrials}(lickL{iTrials} < bhv.RawEvents.Trial{iTrials}.States.MoveSpout(1)) = []; %dont use false licks that occured before spouts were moved in
        lickL{iTrials} = lickL{iTrials} - stimTime(iTrials);
    end
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Port3In') %check for right licks
        lickR{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Port3In;
        lickR{iTrials}(lickR{iTrials} < bhv.RawEvents.Trial{iTrials}.States.MoveSpout(1)) = []; %dont use false licks that occured before spouts were moved in
        lickR{iTrials} = lickR{iTrials} - stimTime(iTrials);
    end
    
    leverIn(iTrials) = min(bhv.RawEvents.Trial{iTrials}.States.Reset(:)) - stimTime(iTrials); %first reset state causes lever to move in
    
    leverTimes = [reshape(bhv.RawEvents.Trial{iTrials}.States.WaitForAnimal1',1,[]) ...
        reshape(bhv.RawEvents.Trial{iTrials}.States.WaitForAnimal2',1,[]) ...
        reshape(bhv.RawEvents.Trial{iTrials}.States.WaitForAnimal3',1,[])];
    
    stimGrab(iTrials) = leverTimes(find(leverTimes == bhv.RawEvents.Trial{iTrials}.States.WaitForCam(1))-1) - stimTime(iTrials); %find start of lever state that triggered stimulus onset
    
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Wire2High') %check for left grabs
        levGrabL{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Wire2High - stimTime(iTrials);
    end
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Wire1High') %check for right grabs
        levGrabR{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Wire1High - stimTime(iTrials);
    end
    
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Wire2Low') %check for left release
        levReleaseL{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Wire2Low - stimTime(iTrials);
    end
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'Wire1Low') %check for right release
        levReleaseR{iTrials} = bhv.RawEvents.Trial{iTrials}.Events.Wire1Low - stimTime(iTrials);
    end
    
    if ~isnan(bhv.RawEvents.Trial{iTrials}.States.Reward(1)) %check for reward state
        water(iTrials) = bhv.RawEvents.Trial{iTrials}.States.Reward(1) - stimTime(iTrials);
    end
end

bhv.ResponseSide(bhv.ResponseSide == 1) = -1; %normalize response side between -1 and 1
bhv.ResponseSide(bhv.ResponseSide == 2) = 1; %normalize response side between -1 and 1

%% build regressors - create design matrix based on event times
motorWindow = (mPreTime + mPostTime) * sRate; %time window to be covered by regressors that describe motor responses (licks or grabs)

%create basic time regressors
timeR = logical(diag(ones(1,frames)));

% allocate cells for other regressors
leverAlignR = cell(1,trialCnt);
stimAlignR = cell(1,trialCnt);

lGrabR = cell(1,trialCnt);
lGrabRelR = cell(1,trialCnt);
rGrabR = cell(1,trialCnt);
rGrabRelR = cell(1,trialCnt);
lLickR = cell(1,trialCnt);
rLickR = cell(1,trialCnt);
leverInR = cell(1,trialCnt);

lVisStimR = cell(1,trialCnt);
rVisStimR = cell(1,trialCnt);
lAudStimR = cell(1,trialCnt);
rAudStimR = cell(1,trialCnt);

visRewardPreStimR = cell(1,trialCnt);
visRewardPostStimR = cell(1,trialCnt);
visPrevRewardPreStimR = cell(1,trialCnt);
visPrevRewardPostStimR = cell(1,trialCnt);

audRewardPreStimR = cell(1,trialCnt);
audRewardPostStimR = cell(1,trialCnt);
audPrevRewardPreStimR = cell(1,trialCnt);
audPrevRewardPostStimR = cell(1,trialCnt);

mixRewardPreStimR = cell(1,trialCnt);
mixRewardPostStimR = cell(1,trialCnt);
mixPrevRewardPreStimR = cell(1,trialCnt);
mixPrevRewardPostStimR = cell(1,trialCnt);

choicePreStimR = cell(1,trialCnt);
choicePostStimR = cell(1,trialCnt);
prevChoicePreStimR = cell(1,trialCnt);
prevChoicePostStimR = cell(1,trialCnt);

waterR = cell(1,trialCnt);

pupilR = cell(1,trialCnt);
faceR = cell(1,trialCnt);
bodyR = cell(1,trialCnt);
piezoR = cell(1,trialCnt);

tic
for iTrials = 1:trialCnt
    % baseline/lever (leverAlignR) and stim/decision (stimAlignR) regressors - these are required to enable variable stimulus onset times
    % time regressor, aligned to start of the (last) lever period. Zero lag regressor after baseline + 1 frame.
    stimOn = floor(bhv.stimOn(iTrials) * sRate); %timepoint of stimulus onset in frames, relative to timepoint zero
    offset = (trialSegments(1)*sRate) + 1 - find(histcounts(stimGrab(iTrials),-preStimDur:1/sRate:postStimDur)); %timepoint of final lever grab in frames
    leverAlignR{iTrials} = false(frames,frames);
    leverAlignR{iTrials}(:,offset+1:end) = timeR(:,1:(frames) - offset);
    leverAlignR{iTrials}((preStimDur*sRate) + stimOn + 1:end,:) = false; %should not extent into stimulus period, defined by current delay value

    stimAlignR{iTrials} = false(frames, postStimDur*sRate);
    stimAlignR{iTrials}(:,1:end-stimOn) = timeR(:,(preStimDur*sRate) + stimOn + 1:end);
       
    %% vis/aud stim
    %regressors cover the last 2s after stimOn
    if bhv.StimType(iTrials) == 1 || bhv.StimType(iTrials) == 3 %visual or mixed stimulus
        if bhv.CorrectSide(iTrials) == 1
            lVisStimR{iTrials} = stimAlignR{iTrials};
            rVisStimR{iTrials} = false(frames, postStimDur * sRate);
        else
            lVisStimR{iTrials} = false(frames, postStimDur * sRate);
            rVisStimR{iTrials} = stimAlignR{iTrials};
        end
    else
        lVisStimR{iTrials} = false(frames, postStimDur * sRate);
        rVisStimR{iTrials} = false(frames, postStimDur * sRate);
    end
    
    if bhv.StimType(iTrials) == 2 || bhv.StimType(iTrials) == 3 %auditory or mixed stimulus
        if bhv.CorrectSide(iTrials) == 1
            lAudStimR{iTrials} = stimAlignR{iTrials};
            rAudStimR{iTrials} = false(frames, postStimDur * sRate);
        else
            lAudStimR{iTrials} = false(frames, postStimDur * sRate);
            rAudStimR{iTrials} = stimAlignR{iTrials};
        end
    else
        lAudStimR{iTrials} = false(frames, postStimDur * sRate);
        rAudStimR{iTrials} = false(frames, postStimDur * sRate);
    end
    
    %% lick regressors
    lLickR{iTrials} = false(frames, motorWindow + 1);
    rLickR{iTrials} = false(frames, motorWindow + 1);
    
    for iRegs = 0 : motorWindow
        licks = lickL{iTrials} - (mPreTime - (iRegs * 1/sRate));
        lLickR{iTrials}(logical(histcounts(licks,-preStimDur:1/sRate:postStimDur)),iRegs+1) = 1;
        
        licks = lickR{iTrials} - (mPreTime - (iRegs * 1/sRate));
        rLickR{iTrials}(logical(histcounts(licks,-preStimDur:1/sRate:postStimDur)),iRegs+1) = 1;
    end
    
    %% lever in
    leverInR{iTrials} = false(frames,frames);
    leverShift = (frames/2) + round(leverIn(iTrials) * sRate); %timepoint in frames were lever moved in, relative to stimOnset
    
    if ~isnan(leverShift)
        if leverShift > 0 %lever moved in during the recorded trial
            leverInR{iTrials}(:, 1: frames - leverShift) = timeR(:, leverShift+1:end);
        else %lever was present before data was recorded
            leverInR{iTrials}(:, abs(leverShift) + 1 : end) = timeR(:,  1: frames + leverShift);
        end
    end

    %% choice and reward
    visRewardPreStimR{iTrials} = (leverAlignR{iTrials} .* (bhv.StimType(iTrials)  == 1)) * (-1 + (2 * bhv.Rewarded(iTrials))); %visual trial, 1 for reward, -1 for no reward
    audRewardPreStimR{iTrials} = (leverAlignR{iTrials} .* (bhv.StimType(iTrials)  == 2)) * (-1 + (2 * bhv.Rewarded(iTrials))); %audio trial, 1 for reward, -1 for no reward
    mixRewardPreStimR{iTrials} = (leverAlignR{iTrials} .* (bhv.StimType(iTrials)  == 3)) * (-1 + (2 * bhv.Rewarded(iTrials))); %audio trial, 1 for reward, -1 for no reward
    
    visRewardPostStimR{iTrials} = (stimAlignR{iTrials} .* (bhv.StimType(iTrials)  == 1)) * (-1 + (2 * bhv.Rewarded(iTrials))); %visual trial, 1 for reward, -1 for no reward
    audRewardPostStimR{iTrials} = (stimAlignR{iTrials} .* (bhv.StimType(iTrials)  == 2)) * (-1 + (2 * bhv.Rewarded(iTrials))); %audio trial, 1 for reward, -1 for no reward
    mixRewardPostStimR{iTrials} = (stimAlignR{iTrials} .* (bhv.StimType(iTrials)  == 3)) * (-1 + (2 * bhv.Rewarded(iTrials))); %audio trial, 1 for reward, -1 for no reward
    
    choicePreStimR{iTrials} = single(leverAlignR{iTrials} .* bhv.ResponseSide(iTrials)); %choice regressor, takes -1 for left and 1 for right responses
    choicePostStimR{iTrials} = single(stimAlignR{iTrials} .* bhv.ResponseSide(iTrials));

    if iTrials == 1 %don't use first trial
        visPrevRewardPreStimR{iTrials} = NaN(size(leverAlignR{iTrials}));
        visPrevRewardPostStimR{iTrials} = NaN(size(stimAlignR{iTrials}));

        audPrevRewardPreStimR{iTrials} = NaN(size(leverAlignR{iTrials}));
        audPrevRewardPostStimR{iTrials} = NaN(size(stimAlignR{iTrials}));
        
        mixPrevRewardPreStimR{iTrials} = NaN(size(leverAlignR{iTrials}));
        mixPrevRewardPostStimR{iTrials} = NaN(size(stimAlignR{iTrials}));
        
        prevChoicePreStimR{iTrials} = NaN(size(leverAlignR{iTrials}));
        prevChoicePostStimR{iTrials} = NaN(size(stimAlignR{iTrials}));
        
    else %for all subsequent trials, use results of last trial to compute previous reward/choice
        visPrevRewardPreStimR{iTrials} = visRewardPreStimR{iTrials - 1};
        visPrevRewardPostStimR{iTrials} = visRewardPostStimR{iTrials - 1};

        audPrevRewardPreStimR{iTrials} = audRewardPreStimR{iTrials - 1};
        audPrevRewardPostStimR{iTrials} = audRewardPostStimR{iTrials - 1};
        
        mixPrevRewardPreStimR{iTrials} = mixRewardPreStimR{iTrials - 1};
        mixPrevRewardPostStimR{iTrials} = mixRewardPostStimR{iTrials - 1};
        
        prevChoicePreStimR{iTrials} = choicePreStimR{iTrials - 1};
        prevChoicePostStimR{iTrials} = choicePostStimR{iTrials - 1};
    end
    
    %determine timepoint of reward given
    waterR{iTrials} = false(frames, sRate);

    if ~isnan(water(iTrials)) && ~isempty(water(iTrials))
        waterOn = (frames/2) + round(water(iTrials) * sRate); %timepoint in frames when reward was given
        waterR{iTrials}(:, 1: size(timeR,2) - waterOn) = timeR(:, waterOn+1:end);
    end
    
    %% lever grab / release
    lGrabR{iTrials} = false(frames, motorWindow + 1);
    lGrabRelR{iTrials} = false(frames, motorWindow + 1);
    
    rGrabR{iTrials} = false(frames, motorWindow + 1);
    rGrabRelR{iTrials} = false(frames, motorWindow + 1);
    
    [grabL,grabRelL] = checkLevergrab(tapDur,postStimDur,levGrabL{iTrials},levReleaseL{iTrials},1/sRate);
    grabL(grabL < -preStimDur) = []; %ensure there are no grab events too early in the trial
    if ~isempty(grabL);grabRelL(grabRelL<grabL(1)) = []; end %make sure there are no release events before first grab
    
    [grabR,grabRelR] = checkLevergrab(tapDur,postStimDur,levGrabR{iTrials},levReleaseR{iTrials},1/sRate);
    grabR(grabR < -preStimDur) = [];
    if ~isempty(grabR); grabRelR(grabRelR<grabR(1)) = []; end
    
    for iRegs = 0 : motorWindow
        
        shiftGrabL = grabL - (mPreTime - (iRegs * 1/sRate));
        shiftGrabR = grabR - (mPreTime - (iRegs * 1/sRate));
        
        shiftGrabRelL = grabRelL - (mPreTime - (iRegs * 1/sRate));
        shiftGrabRelR = grabRelR - (mPreTime - (iRegs * 1/sRate));
        
        if iRegs == 0 %first regressor, find first grab / tap
            
            lGrabR{iTrials}(find(histcounts(shiftGrabL,-preStimDur:1/sRate:postStimDur),1),iRegs+1) = true;
            rGrabR{iTrials}(find(histcounts(shiftGrabR,-preStimDur:1/sRate:postStimDur),1),iRegs+1) = true;
            
        else
            
            regOut = leverEvents(iRegs, grabRelL, shiftGrabL, find(lGrabR{iTrials}(:,iRegs)), -preStimDur:1/sRate:postStimDur, mPreTime*sRate, 'pre'); %code to compute events for current lever regressor
            regOut(regOut > frames) = [];
            lGrabR{iTrials}(regOut,iRegs+1) = true;
            
            regOut = leverEvents(iRegs, grabRelR, shiftGrabR, find(rGrabR{iTrials}(:,iRegs)), -preStimDur:1/sRate:postStimDur, mPreTime*sRate, 'pre'); %code to compute events for current lever regressor
            regOut(regOut > frames) = [];
            rGrabR{iTrials}(regOut,iRegs+1) = true;
            
            regOut = leverEvents(iRegs, grabL, shiftGrabRelL, find(lGrabRelR{iTrials}(:,iRegs)), -preStimDur:1/sRate:postStimDur, mPreTime*sRate, ''); %code to compute events for current lever regressor
            regOut(regOut > frames) = [];
            lGrabRelR{iTrials}(regOut,iRegs+1) = true;
            
            regOut = leverEvents(iRegs, grabR, shiftGrabRelR, find(rGrabRelR{iTrials}(:,iRegs)), -preStimDur:1/sRate:postStimDur, mPreTime*sRate, ''); %code to compute events for current lever regressor
            regOut(regOut > frames) = [];
            rGrabRelR{iTrials}(regOut,iRegs+1) = true;           
            
        end
    end
    
    %% pupil/video regressors
    frameFile = dir([cPath 'BehaviorVideo' filesep '*frameTimes_'  num2str(trials(iTrials),'%04i') '*_1.mat']);
    if exist([cPath 'BehaviorVideo' filesep 'faceVars_' int2str(trials(iTrials)) '.mat'],'file') ~= 2 || isempty(frameFile)  %check if files exists on hdd and pull from server otherwise
        if exist([cPath 'BehaviorVideo' filesep]) ~= 2
            mkdir([cPath 'BehaviorVideo' filesep]);
        end
        frameFile = dir([sPath 'BehaviorVideo' filesep '*frameTimes_'  num2str(trials(iTrials),'%04i') '*_1.mat']);
        
        %get results from pupil analysis
        copyfile([sPath filesep 'BehaviorVideo' filesep 'faceVars_' int2str(trials(iTrials)) '.mat'],[cPath filesep 'BehaviorVideo' filesep 'faceVars_' int2str(trials(iTrials)) '.mat']);
        % copy frametimes from server to local
        copyfile([sPath filesep 'BehaviorVideo' filesep frameFile.name],[cPath filesep 'BehaviorVideo' filesep frameFile.name]);
    end
    
    load([cPath 'BehaviorVideo' filesep 'faceVars_' int2str(trials(iTrials)) '.mat'])
    load([cPath 'BehaviorVideo' filesep frameFile.name])
    
    %absolute trial onset time for bpod - use to synchronize video to behavioral events
    trialOn = bhv.TrialStartTime(iTrials) * 86400 + (stimTime(iTrials) - preStimDur);
    frameTimes = (frameTimes  * 86400) - trialOn;
    trialOn = find(frameTimes > 0,1); %trial on in frames
    bhvFrameRate = round(1/median(diff(frameTimes)));
    trialOn = round(trialOn * (sRate / bhvFrameRate)); %trial on in resampled frames
    
    %resample to match imaging data framerate
    try
        pupil = mean(eyeVars.axes,2); %pupil diameter
        idx = zscore(diff(pupil)) > 0.5; %index for blinks
        pupil(idx) = mean(pupil); %replace blinks with pupil average
        pupil = [repmat(pupil(1),21,1); pupil; repmat(pupil(end),21,1)]; %add some padding on both sides to avoid edge effects when resampling
        pupil = resample(pupil, sRate, bhvFrameRate); %resample to match imaging data framerate
        offset = ceil(21 * sRate / bhvFrameRate); %required offset to remove padding after resampling
        pupil = pupil(offset : end - offset); %remove padds
        pupil = smooth(pupil,'rlowess'); %do some smoothing
        idx = trialOn:trialOn + (frames - 1); %index for data that matches the widefield
        pupilR{iTrials} = [single(pupil(idx-bhvOffset)) single(pupil(idx)) single(pupil(idx+bhvOffset))]; %only use trial-relevant frames
    catch
        pupilR{iTrials} = NaN(frames,3,'single'); %can't use this trial
    end
    
    %get svd data from cam1 - this is usually the one that sees the animals face
    trialOn = bhv.TrialStartTime(iTrials) * 86400 + (stimTime(iTrials) - preStimDur);
    cTimes = frameTimes1 * 86400 - trialOn;
    trialOn = find(cTimes > 0,1); %trial on in frames
    bhvFrameRate = round(1/median(diff(cTimes)));
    idx = trialOn : trialOn + ceil(frames*(bhvFrameRate / sRate)); %index for data that matches the widefield after resampling
    faceMotion = [V1(idx-bhvOffset, 1:bhvDimCnt1) V1(idx, 1:bhvDimCnt1) V1(idx+bhvOffset, 1:bhvDimCnt1)];
    try
        faceMotion = [repmat(faceMotion(1,:),21,1); faceMotion; repmat(faceMotion(end,:),21,1)]; %add some padding on both sides to avoid edge effects when resampling
        faceMotion = resample(double(faceMotion), sRate, bhvFrameRate); %resample to match imaging data framerate
        faceMotion = conv2(faceMotion',ones(1,smth)/smth,'same')'; %smooth trace with moving average of 'smth' points
        faceMotion = faceMotion(offset : end - offset,:); %remove padds
        faceR{iTrials} = single(faceMotion(1:frames,:)); %only use trial-relevant frames
    catch
        faceR{iTrials} = NaN(frames,bhvDimCnt1*3, 'single'); %can't use this trial
    end
    
    %get svd data from cam2 - this is usually the one that sees the animal from the bottom
    trialOn = bhv.TrialStartTime(iTrials) * 86400 + (stimTime(iTrials) - preStimDur);
    cTimes = frameTimes2 * 86400 - trialOn;
    trialOn = find(cTimes > 0,1); %trial on in frames
    bhvFrameRate = round(1/median(diff(cTimes)));
    idx = trialOn : trialOn + ceil(frames*(bhvFrameRate / sRate)); %index for data that matches the widefield after resampling
    bodyMotion = [V2(idx-bhvOffset, 1:bhvDimCnt1) V2(idx, 1:bhvDimCnt1) V2(idx+bhvOffset, 1:bhvDimCnt1)];    
    try
        bodyMotion = [repmat(bodyMotion(1,:),21,1); bodyMotion; repmat(bodyMotion(end,:),21,1)]; %add some padding on both sides to avoid edge effects when resampling
        bodyMotion = resample(double(bodyMotion), sRate, bhvFrameRate); %resample to match imaging data framerate
        bodyMotion = conv2(bodyMotion',ones(1,smth)/smth,'same')'; %smooth trace with moving average of 'smth' points
        bodyMotion = bodyMotion(offset : end - offset,:); %remove padds
        bodyR{iTrials} = single(bodyMotion(1:size(timeR,1),:)); %only use trial-relevant frames
    catch
        bodyR{iTrials} = NaN(frames, bhvDimCnt1*3, 'single'); %can't use this trial
    end
    
    %% piezo sensor information
    if exist([cPath 'Analog_'  num2str(trials(iTrials)) '.dat'],'file') ~= 2  %check if files exists on hdd and pull from server otherwise
        cFile = dir([sPath 'Analog_'  num2str(trials(iTrials)) '.dat']);
        copyfile([sPath 'Analog_'  num2str(trials(iTrials)) '.dat'],[cPath 'Analog_'  num2str(trials(iTrials)) '.dat']);
    end
    
    [~,Analog] = Widefield_LoadData([cPath 'Analog_'  num2str(trials(iTrials)) '.dat'],'Analog'); %load analog data    
    stimOn = find(diff(double(Analog(stimLine,:)) > 1500) == 1); %find stimulus onset in current trial
    ind = find((find(diff(double(Analog(stimLine,:)) > 1500) == -1) - stimOn) > 2,1); %only use triggers that are more than 2ms long
    stimOn = stimOn(ind) + 1;
    piezoR{iTrials} = Analog(piezoLine,stimOn - (preStimDur * 1000) : stimOn + (postStimDur * 1000) - 1); % data from piezo sensor. Should encode animals hindlimb motion.
        
    % give some feedback over progress
    if rem(iTrials,50) == 0
        fprintf(1, 'Current trial is %d out of %d\n', iTrials,trialCnt);
        toc
    end
end

%% rebuild piezo sensor to get proper design matrix
pTrace = double(cat(2,piezoR{:}));
pTrace = smooth(pTrace ./ std(pTrace),100);
pTrace = pTrace > 0.5; %threshold normalized sensor data
pTrace = diff([0;pTrace]) == 1; %find event onsets
pTrace = logical(histcounts(find(pTrace), 0:1000/sRate:length(pTrace)))'; %resample to imaging frame rate. This is the zero lag regressor.
pTrace = reshape(pTrace,[],trialCnt);

idx = [-((mPreTime * sRate): -1 : 1) 0 (1:(mPostTime * sRate))]; %index for design matrix to cover pre- and post motor action
for iTrials = 1:trialCnt
    cIdx = find(pTrace(:,iTrials)) + idx;
    cIdx(cIdx < 1) = 0;
    cIdx(cIdx > frames) = frames;
    cIdx = cIdx + (0:frames:frames*length(idx)-1);
    cIdx(cIdx < 1) = frames;
    cIdx(cIdx > (frames * length(idx))) = frames * length(idx);
    
    piezoR{iTrials} = false(frames, length(idx));
    piezoR{iTrials}(cIdx(:)) = true;
    piezoR{iTrials}(end,:) = false; %don't use last timepoint of design matrix to avoid confusion with indexing.
    
end
clear pTrace idx cIdx

%% reshape regressors, make design matrix and indices for regressors that are used for the model
leverAlignR = cat(1,leverAlignR{:});
stimAlignR = cat(1,stimAlignR{:});

lGrabR = cat(1,lGrabR{:});
lGrabRelR = cat(1,lGrabRelR{:});
rGrabR = cat(1,rGrabR{:});
rGrabRelR = cat(1,rGrabRelR{:});

lLickR = cat(1,lLickR{:});
rLickR = cat(1,rLickR{:});
leverInR = cat(1,leverInR{:});

lVisStimR = cat(1,lVisStimR{:});
rVisStimR = cat(1,rVisStimR{:});
lAudStimR = cat(1,lAudStimR{:});
rAudStimR = cat(1,rAudStimR{:});

visRewardPreStimR = cat(1,visRewardPreStimR{:});
visRewardPostStimR = cat(1,visRewardPostStimR{:});
audRewardPreStimR = cat(1,audRewardPreStimR{:});
audRewardPostStimR = cat(1,audRewardPostStimR{:});
mixRewardPreStimR = cat(1,mixRewardPreStimR{:});
mixRewardPostStimR = cat(1,mixRewardPostStimR{:});

visPrevRewardPreStimR = cat(1,visPrevRewardPreStimR{:});
visPrevRewardPostStimR = cat(1,visPrevRewardPostStimR{:});
audPrevRewardPreStimR = cat(1,audPrevRewardPreStimR{:});
audPrevRewardPostStimR = cat(1,audPrevRewardPostStimR{:});
mixPrevRewardPreStimR = cat(1,mixPrevRewardPreStimR{:});
mixPrevRewardPostStimR = cat(1,mixPrevRewardPostStimR{:});

choicePreStimR = cat(1,choicePreStimR{:});
choicePostStimR = cat(1,choicePostStimR{:});
prevChoicePreStimR = cat(1,prevChoicePreStimR{:});
prevChoicePostStimR = cat(1,prevChoicePostStimR{:});

waterR = cat(1,waterR{:});
piezoR = cat(1,piezoR{:});

pupilR = cat(1,pupilR{:});
camIdx = ~isnan(pupilR(:,1)); %index for trials where bhv frames were available
pupilR(camIdx,:) = zscore(pupilR(camIdx,:));

%% combine body and face data and use svd to merge into one set of predictors (to reduce redundancy between video regressors)
videoR = [cat(1,faceR{:}) cat(1,bodyR{:})]; clear faceR bodyR
videoR = videoR(camIdx,:);

idx = [1:size(videoR,2)/6 size(videoR,2)/2 + 1 : size(videoR,2)/2 + size(videoR,2)/6]; %index for first set of regressors for both cameras (shifted back in time)
pastVidR = videoR(:,idx);
presentVidR = videoR(:,idx+size(videoR,2)/6);
futureVidR = videoR(:,idx+size(videoR,2)/3);
clear videoR

[pastVidV, mPastVidV, stdPastVidV, pastVidU] = compressVideoRegs(pastVidR,bhvDimCnt2); %compress into one set of regress and zscore so it can be used in GLM
pastVidR = NaN(length(camIdx),bhvDimCnt2,'single');
pastVidR(camIdx,:) = pastVidV; clear pastVidV
save([cPath filesep 'pastVid.mat'], 'pastVidR', 'mPastVidV' ,'stdPastVidV', 'pastVidU');

[presentVidV, mPresentVidV, stdPresentVidV, presentVidU] = compressVideoRegs(presentVidR,bhvDimCnt2); %compress into one set of regress and zscore so it can be used in GLM
presentVidR = NaN(length(camIdx),bhvDimCnt2,'single');
presentVidR(camIdx,:) = presentVidV; clear pastVidV
save([cPath filesep 'presentVid.mat'], 'presentVidR', 'mPresentVidV' ,'stdPresentVidV', 'presentVidU');

[futureVidV, mFutureVidV, stdFutureVidV, futureVidU] = compressVideoRegs(futureVidR,bhvDimCnt2); %compress into one set of regress and zscore so it can be used in GLM
futureVidR = NaN(length(camIdx),bhvDimCnt2,'single');
futureVidR(camIdx,:) = futureVidV; clear pastVidV
save([cPath filesep 'futureVid.mat'], 'futureVidR', 'mFutureVidV' ,'stdFutureVidV', 'futureVidU');


%% create full design matrix
fullR = [leverAlignR stimAlignR lGrabR lGrabRelR rGrabR rGrabRelR lLickR rLickR ...
    leverInR lVisStimR rVisStimR lAudStimR rAudStimR visRewardPreStimR visRewardPostStimR ...
    audRewardPreStimR audRewardPostStimR mixRewardPreStimR mixRewardPostStimR ... 
    visPrevRewardPreStimR visPrevRewardPostStimR audPrevRewardPreStimR audPrevRewardPostStimR ...
    mixPrevRewardPreStimR mixPrevRewardPostStimR choicePreStimR choicePostStimR ...
    prevChoicePreStimR prevChoicePostStimR waterR pupilR pastVidR presentVidR futureVidR piezoR];

trialIdx = isnan(mean(fullR,2)); %don't use first trial or trials that failed to contain behavioral video data
fullR(trialIdx,:) = []; %clear bad trials
fprintf(1, 'Rejected %d/%d trials for NaN regressors\n', sum(trialIdx) / frames,trialCnt);

idx = nansum(abs(fullR)) < 10; %reject regressors that are too sparse
fullR(:,idx) = []; %clear empty regressors
fprintf(1, 'Rejected %d/%d empty regressors\n', sum(idx),length(idx));

%index to reconstruct different response kernels
recIdx = [ones(1,size(leverAlignR,2)) ones(1,size(stimAlignR,2))*2 ones(1,size(lGrabR,2))*3 ones(1,size(lGrabRelR,2))*4 ones(1,size(rGrabR,2))*5 ...
    ones(1,size(rGrabRelR,2))*6 ones(1,size(lLickR,2))*7 ones(1,size(rLickR,2))*8 ones(1,size(leverInR,2))*9 ...
    ones(1,size(lVisStimR,2))*10 ones(1,size(rVisStimR,2))*11 ones(1,size(lAudStimR,2))*12 ones(1,size(rAudStimR,2))*13 ...
    ones(1,size(visRewardPreStimR,2))*14 ones(1,size(visRewardPostStimR,2))*15 ones(1,size(audRewardPreStimR,2))*16 ones(1,size(audRewardPostStimR,2))*17 ...
    ones(1,size(mixRewardPreStimR,2))*18 ones(1,size(mixRewardPostStimR,2))*19 ones(1,size(visPrevRewardPreStimR,2))*20 ones(1,size(visPrevRewardPostStimR,2))*21 ...
    ones(1,size(audPrevRewardPreStimR,2))*22 ones(1,size(audPrevRewardPostStimR,2))*23 ones(1,size(mixPrevRewardPreStimR,2))*24 ones(1,size(mixPrevRewardPostStimR,2))*25 ...
    ones(1,size(choicePreStimR,2))*26 ones(1,size(choicePostStimR,2))*27 ones(1,size(prevChoicePreStimR,2))*28 ones(1,size(prevChoicePostStimR,2))*29 ...
    ones(1,size(waterR,2))*30 ones(1,size(pupilR,2))*31 ones(1,size(pastVidR,2))*32 ones(1,size(presentVidR,2))*33 ones(1,size(futureVidR,2))*34 ones(1,size(piezoR,2))*35]; 

recLabels = {
    'leverAlign' 'stimAlign' 'lGrab' 'lGrabRel' 'rGrab' 'rGrabRel' 'lLick' 'rLick' 'leverIn' 'lVisStim' 'rVistStim' ...
    'lAudStim' 'rAudStim' 'visReward' 'audReward' 'mixReward' 'visPrevReward' 'audPrevReward' 'mixPrevReward' 'choice' ...
    'prevChoice' 'water' 'pupil' 'pastVidR' 'presentVidR' 'futureVidR' 'piezoR'};

%clear individual regressors
clear leverAlignR stimAlignR lGrabR lGrabRelR rGrabR rGrabRelR waterR lLickR rLickR leverInR ...
      lVisStimR rVisStimR lAudStimR rAudStimR visRewardR audRewardR mixRewardR visPrevRewardR audPrevRewardR ... 
       mixPrevRewardR choiceR prevChoiceR pupilR pastVidR presentVidR futureVidR piezoR

% save some results
save([cPath filesep 'regData.mat'], 'fullR', 'idx' ,'trialIdx', 'recIdx', 'recLabels');

%% run ridge regression in low-D and find error term
U = arrayShrink(U,mask);
Vc = reshape(Vc,dims,[]);
Vc(:,trialIdx) = [];
Vc = bsxfun(@minus, Vc, mean(Vc, 2)); %make sure Vc is zero-mean

if exist([cPath 'ridgeTest.mat']) == 2 && ~newRun %load ridgetest file to get correct ridge penalty
    load([cPath 'ridgeTest.mat'])
else
    options = optimset('PlotFcns',@optimplotfval);
    options.TolX = 10;
    [ridgeVal,ridgeError,exitflag,searchOutput] = fminsearch(@(u) ridgeFastPredict(Vc, fullR, ridgeFolds, u), ridgeVal, options);
    save([cPath 'ridgeTest.mat'], 'ridgeVal', 'ridgeError','exitFlag','searchOutput')
end
fprintf('Selected ridge penalty: %f, Prediction RMSE: %f\n', ridgeVal, ridgeError);

%% run regression in low-D
[dimBeta, dimBetaCI] = ridgeFast(Vc', fullR, ridgeVal); %perform ridge regression to get beta weights and confidence intervals
dimBetaCI = squeeze(dimBetaCI(:,2,:)) - dimBeta;
save([cPath 'dimBeta.mat'], 'dimBeta')
save([cPath 'dimBetaCI.mat'], 'dimBetaCI')

beta = dimBeta * U'; %compute beta weight maps
betaCI = dimBetaCI * U'; %compute beta confidence maps

parBeta = cell(1,size(recLabels,2));
parBetaCI = cell(1,size(recLabels,2));
for iModels = 1:size(recLabels,2)
    cIdx = recIdx(~idx) == iModels;
    [parBeta{iModels}, parBetaCI{iModels}] = ridgeFast(Vc', fullR, ridgeVal); %perform ridge regression to get model error
    parBetaCI{iModels} = squeeze(parBetaCI{iModels}(:,2,:)) - parBeta{iModels};
end
save([cPath 'parBeta.mat'], 'parBeta')
save([cPath 'parBetaCI.mat'], 'parBetaCI')

%% compute correlations in low-D
trialSegments = trialSegments * sRate;
if exist([cPath 'dimCorrFull.mat']) == 2 && ~newRun
    if ~isunix
        load([cPath 'dimCorrFull.mat'])
        figure;
        imagesc(arrayShrink(dimCorr.^2, mask, 'split')); axis square; colorbar; colormap jet
        title('R^2 - Dimensional full model')
    end
else
    Vm = (fullR * dimBeta)';
    dimCorrFull = matCorr(Vc, Vm, U); %compute correlation matrix between data and model estimate
    save([cPath 'dimCorrFull.mat'], 'dimCorrFull')
    
    for iModels = 1:length(parBeta)  %compute correlations for partial models
        parIdx = recIdx(~idx) == iModels;
        parCorrFull{iModels} = matCorr(Vc, (fullR(:,parIdx)*parBeta{iModels}(parIdx,:))', U); %compute correlation matrix between data and model estimate
    end
    save([cPath 'parCorrFull.mat'], 'parCorrFull', 'recLabels')
    
    dimCorrSeg = zeros(size(U,1),length(trialSegments));
    for iSegments = 1:length(trialSegments)
        % Get index for all frames that match the selected trial segments.
        % 1:40 baseline, 41:80 lever grab, 81:120 stimulus, 121:160 decision from each trial.
        a = ((0 : size(fullR,1) / sum(trialSegments) - 1) * sum(trialSegments));
        b = [0 cumsum(trialSegments)];
        cIdx = bsxfun(@plus,repmat(b(iSegments)+1:b(iSegments+1),length(a),1), a');
        cIdx = reshape(cIdx',1,[]);
        
        dimCorrSeg(:,iSegments) = matCorr(Vc(:,cIdx), Vm(:,cIdx), U); %compute correlation matrix between data and model estimate
        
        for iModels = 1:length(parBeta)  %compute correlations for partial models
            parIdx = (recIdx(~idx) == iModels); %index for current regressors
            parCorrSeg{iModels}(:,iSegments) = matCorr(Vc(:,cIdx), (fullR(cIdx,parIdx)*parBeta{iModels}(parIdx,:))', U); %compute correlation matrix between data and model estimate
        end
    end
    save([cPath 'dimCorrSeg.mat'], 'dimCorrSeg')
    save([cPath 'parCorrSeg.mat'], 'parCorrSeg', 'recLabels')
end

%% collect beta maps and reconstruct kernels
beta = arrayShrink(gather(beta)',mask,'split'); %spatially reconstruct beta maps
fullBeta = NaN(size(beta,1),size(beta,2),length(idx),'single');
fullBeta(:,:,~idx) = beta; %full matrix to include regressors that were rejected before

betaCI = arrayShrink(gather(betaCI)',mask,'split'); %spatially reconstruct beta confidence maps
fullBetaCI = NaN(size(betaCI,1),size(betaCI,2),length(idx),'single');
fullBetaCI(:,:,~idx) = betaCI;

 %save individual regressor sets
for iRegs = 1:length(recLabels)
    data = fullBeta(:,:,recIdx == iRegs);
    save([cPath recLabels{iRegs} 'B.mat'], 'data', 'iRegs', 'recLabels','-v7.3');
    clear data
end

%save full regressor set
save([cPath 'fullBeta.mat'], 'fullBeta', 'recIdx', 'recLabels','-v7.3');
save([cPath 'fullBetaCI.mat'], 'fullBetaCI', 'recIdx', 'recLabels','-v7.3');



%% nested functions
function regOut = leverEvents(cReg, times, shiftTimes, lastTimes, timeFrame, preTimeShift, dMode) %code to compute events for current lever regressor

regOut = [];
if cReg == 0 %first regressor, find first grab
    regOut = find(histcounts(shiftTimes,timeFrame),1);
    
elseif cReg < preTimeShift && strcmpi(dMode,'pre') %negative shift regressors - only for grab or tap
    regOut = lastTimes + 1; %find last event and shift one forward
    if isempty(regOut)
        regOut = find(histcounts(shiftTimes,timeFrame),1); %check if new event can be found if none is present so far
    end
    
elseif cReg == preTimeShift %this is the zero-lag regressor. use all available events
    regOut = find(histcounts(shiftTimes,timeFrame)); %times and shifttimes should be the same here.
    
elseif cReg > preTimeShift %for positive shift, use the last regressor but eliminate event if there is overlap with zero-lag regressor.
    regOut = lastTimes + 1; %find last event and shift one forward
    if ~isempty(regOut)
        regOut(ismember(regOut, find(histcounts(times,timeFrame)))) = []; %remove events that overlap with the given zero-lag regressor. Can also be from a different event type like handle release.
    end
end



function [grabOn,grabRel,tapOn,tapRel] = checkLevergrab(tapDur,postStimDur,grabs,release,minTime)

grabOn = [];
grabRel = [];
tapOn = [];
tapRel = [];

if ~isempty(grabs)
    Cnt = 0;
    grabCnt = 0;
    tapCnt = 0;
    
    while true
        Cnt = Cnt +1;
        if length(grabs) < Cnt %no more grabs
            break;
        end
        
        idx = find(release > grabs(Cnt),1);
        if ~isempty(idx) %if there is a release that follows current grab
            cGrab = release(idx) - grabs(Cnt); %duration of current grab
        else
            cGrab = postStimDur - grabs(Cnt);
        end
        
        %% more grabs available - check start time of next grab and merge if they are very close in time
        if length(grabs) > Cnt && ~isempty(idx)
            while (grabs(Cnt+1) - release(idx)) <= minTime %time diff between grabs is less than minimum duration. Merge with next grab.
                
                release(idx) = []; %delete current release
                grabs(Cnt+1) = []; %delete next grab
                
                idx = find(release > grabs(Cnt),1); %next release
                if ~isempty(idx) %if there is a release that follows current grab
                    cGrab = release(idx) - grabs(Cnt); %duration of current grab
                else
                    cGrab = postStimDur - grabs(Cnt);
                end
                
                if length(grabs) <= Cnt %no more grabs
                    break
                end
                
            end
        end
        
        %% check if current grab is grab or tap
        if cGrab <= tapDur
            tapCnt = tapCnt + 1;
            tapOn(tapCnt) = grabs(Cnt);
            if ~isempty(idx)
                tapRel(tapCnt) = idx;
            end
        else
            grabCnt = grabCnt + 1;
            grabOn(grabCnt) = grabs(Cnt);
            if ~isempty(idx)
                grabRel(grabCnt) = release(idx);
            end
        end
        
        if isempty(idx) || length(grabs) <= Cnt %no more grabs/releases
            break;
        end
    end
end

function dimCorr = matCorr(Vc, Vm, U)
% compute correlation between predicted and real imaging data
% Vc are real temporal components, Vm is modeled. U are spatial components

% Vm = (fullR * dimBeta)';

covVc = cov(Vc');  % S x S
covVm = cov(Vm');  % S x S
Vm = bsxfun(@minus, Vm, mean(Vm,2));
cCovV = Vm * Vc' / (size(Vc, 2) - 1);  % S x S
covP = sum((U * cCovV) .* U, 2)';  % 1 x P
varP1 = sum((U * covVc) .* U, 2)';  % 1 x P
varP2 = sum((U * covVm) .* U, 2)';  % 1 x P
stdPxPy = varP1 .^ 0.5 .* varP2 .^ 0.5; % 1 x P
dimCorr = (covP ./ stdPxPy)';

function [vidV, meanV, stdV, vidU] = compressVideoRegs(vidR,dimCnt)
% perform svd to further compres video regressors and reduce redundancy
% between cameras

[vidV,S,vidU]  = svd(vidR, 'econ');
vidU = vidU(:,1:dimCnt)'; %the resulting U is nSVD x regs 
vidV = vidV*S; %convolve V and S to stick to the usual analysis. This is frames x nSVD and is used as regressor in the GLM.
vidV = vidV(:,1:dimCnt); %get request nr of regressors

%zscore video regressor, return mean and std for later reconstruction
meanV = mean(vidV);
stdV = std(vidV);
vidV = bsxfun(@minus,vidV,meanV);
vidV = bsxfun(@rdivide,vidV,stdV);