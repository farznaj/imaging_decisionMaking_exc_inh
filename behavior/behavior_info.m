% this script is no more needed since u have the following useful functions.
% [outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp);
% [~, ~, stimrate, stimtype] = setStimRateType(alldata); % stimtype = [multisens, onlyvis, onlyaud];


%% set some params related to behavior and trial types.

[nevents, stimdur, stimrate, stimtype] = setStimRateType(alldata);
cb = alldata(1).categoryBoundaryHz;
% lowRateTrials = stimrate < cb;
% highRateTrials = stimrate > cb;

%
allResp = [alldata.responseSideIndex];
leftChoiceTrials = (allResp==1);
rightChoiceTrials = (allResp==2);

%
outcomes = [alldata.outcome];
outcomes(trs2rmv) = NaN;

%
correctL = (outcomes==1) & (allResp==1);
correctR = (outcomes==1) & (allResp==2);
numCorr_L_R = [sum(correctL), sum(correctR)]

%
incorrectL = (outcomes==0) & (allResp==1);
incorrectR = (outcomes==0) & (allResp==2);

numIncorr_L_R = [sum(incorrectL), sum(incorrectR)]


%%
% trials that the mouse entered the allow correction state. (remember :
% this does not give all trials that the mouse 1st committed error. If the mouse was on sideChoose, he will go to punish (and not punish allow correction).)
a = arrayfun(@(x)x.parsedEvents.states.punish_allowcorrection, alldata, 'uniformoutput', 0);
allowCorrectEntered = ~cellfun(@isempty, a);

% trials that mouse licked the error side during the decision time. Afterwards, the mouse may have committed it
% (hence entered allow correction or punish) or may have licked the correct side.
a = arrayfun(@(x)x.parsedEvents.states.errorlick_again_wait, alldata, 'uniformoutput', 0);
errorlick_again_wait_entered = (~cellfun(@isempty, a));

% trials that mouse licked the correct side during the decision time. Afterwards, the mouse may have committed it
% (hence reward) or may have licked the error side.
a = arrayfun(@(x)x.parsedEvents.states.correctlick_again_wait, alldata, 'uniformoutput', 0);
correctlick_again_wait_entered = (~cellfun(@isempty, a));


%%
correctButAllowCorr = (allowCorrectEntered & outcomes==1); 
% outcomes(allowCorrectEntered & outcomes==1) = 0; % set to failure the outcome of correct trials on which the mouse used allow correction (bc we want to consider the outcome of the first choice)
correctBut1stUncommitError = (errorlick_again_wait_entered & outcomes==1); % all trials that ended up correct but were preceded by at least one incorrect lick.

incorrectBut1stUncommitCorrect = (correctlick_again_wait_entered & outcomes==0); % all trials that ended up incorrect but were preceded by at least one correct lick.



