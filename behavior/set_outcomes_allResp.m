function [outcomes, allResp, allResp_HR_LR] = set_outcomes_allResp(alldata, uncommittedResp, allowCorrectResp, allowCorrectOutcomeChange)
%
% set outcome and response side for each trial, taking into account
% allcorrection and uncommitted responses.
%
% OUTPUTS:
% outcomes: outcome of each trial:
%    1: success, 0: failure, -1: early decision, -2: no decision, -3: wrong initiation,
%   -4: no center commit, -5: no side commit
% allResp: response side of each trial: 1: left, 2: right.
% allResp_HR_LR: response side based on the contingency --> 1 for HR choice, 0 for LR choice.
%
% INPUTS:
% uncommittedResp: optional, could be: 'remove', 'change' or 'nothing'
% (default): how to deal with trials that mouse 1st made an uncommitted
% lick and then switched the response to the other side.
%
% allowCorrectResp: optional, could be 'remove', 'nothing', or 'change'
% (default): how to deal with trials that mouse entered allowCorrection
% state. Default changes the outcome and response side to the original lick
% (as if mouse was not allowed to correct).
%
% allowCorrectOutcomeChange: optional % only effective when
% allowCorrectResp is set to 'change'. If 0, outcome of allowCorrEntered
% trials wont be changed (although animal's choice will be changed). If 1,
% outcome will be changed as well.


%%
if ~exist('uncommittedResp', 'var')
    uncommittedResp = 'nothing'; % leave outcome and allResp as they are (ie go with the final choice of the mouse).
end

if ~exist('allowCorrectResp', 'var')
    allowCorrectResp = 'change'; % change the response on trials that entered allowCorrection to the original choice.
end

if ~exist('allowCorrectOutcomeChange', 'var')
    allowCorrectOutcomeChange = 1;
    % only effective when allowCorrectResp is 'change'. If 0, outcome of
    % allowCorrEntered trials wont be changed (although animal's choice
    % will be changed). If 1, outcome will be changed as well.
end


%% Set outcomes and allResp.

outcomes = [alldata.outcome];
allResp = [alldata.responseSideIndex]; % Note: it will be NaN for any outcome other than success and failure.


%% Take care of outcome and allResp of allowCorrection trials.

% trials that the mouse entered the allow correction state. (remember :
% this does not give all trials that the mouse 1st committed error. If the mouse was on sideChoose, he will go to punish (and not punish allow correction).)
a = arrayfun(@(x)x.parsedEvents.states.punish_allowcorrection, alldata, 'uniformoutput', 0);
allowCorrectEntered = ~cellfun(@isempty, a);
fprintf('%.3f = Fraction of trials aninmal entered allowCorrection\n', nanmean(allowCorrectEntered))

if sum(allowCorrectEntered) > 0
    a = nanmean(allowCorrectEntered & outcomes==1) / nanmean(allowCorrectEntered);
    fprintf('%.3f = Fract of allowCorrectEntered trials with final success\n', a)
    a = nanmean(allowCorrectEntered & outcomes==0) / nanmean(allowCorrectEntered);
    fprintf('%.3f = Fract of allowCorrectEntered trials with final failure\n', a)
    a = nanmean(allowCorrectEntered & outcomes==-5) / nanmean(allowCorrectEntered);
    fprintf('%.3f = Fract of allowCorrectEntered trials with final no sideLickAgain\n', a)
    a = nanmean(allowCorrectEntered & outcomes==-2) / nanmean(allowCorrectEntered);
    fprintf('%.3f = Fract of allowCorrectEntered trials with final noChoice. These are really noSideLickAgain, but must correspond to earlier training days when you did not have the state wait4decision2. \n', a)
    
    if nanmean(ismember(unique(outcomes(allowCorrectEntered)), [-5 -2 0 1])) ~=1
        error('This cannot happen!')
    end
end

% sum(allowCorrectEntered & outcomes==0) % cases that animal entered allowCorrection and yet didn't change his choice.
% sum(allowCorrectEntered & outcomes==1) % animal entered allowCorrection but changed his choice later.
% sum(allowCorrectEntered & outcomes==-5) % animal entered allowCorrection but didn't side lick again eventually.
% sum(allowCorrectEntered & outcomes==-2) % animal entered allowCorrection, but didn't side lick again eventually. This case is only
% for earlier training days of your early mice. See the comment below.

% Remember at first you did not have wait4decision2 and it was all
% wait4decision. What it means is that you didn't distinguish between
% animal not choosing at all (outcome=-2) and not side licking again
% (outcome=-5). So you may have the following case for the earlier training
% days of some of your early mice: (allowCorrectEntered & outcomes==-2). [In
% reality when a mouse enters allowCorrection, then the outcome cannot be
% -2 (bc he has made a choice) but this only happened because you did not
% distinguish between no choice and no side lick again at earlier days of
% your paradigm.]


switch allowCorrectResp
    case 'remove'
        disp('Removing allCorrectEntered trials!')
        allResp(allowCorrectEntered) = NaN;
        outcomes(allowCorrectEntered) = NaN;
        
    case 'change'
        aa = allResp(allowCorrectEntered & outcomes==1); % change to the original choice the response on trials that mouse entered allow correction and then changed his choice.
        a = aa;
        a(aa==1) = 2;
        a(aa==2) = 1;
        allResp(allowCorrectEntered & outcomes==1) = a;
        
        
        for itr = find(allowCorrectEntered & ismember(outcomes, [-2, -5])) % mouse didn't side-lick-again at the end, set the resp side to the original choice. % if wondering why outcome=-2 is included, read your comments above. % find(allowCorrectEntered & outcomes==-5)
            pat = alldata(itr).parsedEvents.states.punish_allowcorrection(1);
            
            if ismember(pat, alldata(itr).parsedEvents.pokes.L)
                allResp(itr) = 1; % left
            elseif ismember(pat, alldata(itr).parsedEvents.pokes.R)
                allResp(itr) = 2; % right
            end
        end
        
        if sum(allowCorrectEntered) > 0
            if allowCorrectOutcomeChange % change both outcome and allResp
                disp('Changing the response and outcome of allCorrectEntered trials!')
                outcomes(allowCorrectEntered) = 0; % their outcome can be 0, 1, or -5, but u're going with mouse's 1st choice so you prefer this to outcomes(allowCorrectEntered & outcomes==1) = 0;
            else % don't change outcome, only change allResp
                disp('Changing the response BUT NOT outcome of allCorrectEntered trials!')
            end
        end
        
        
    case 'nothing'
        disp('allCorrectEntered trials left unchanged!')
end


%{
if size(alldata(itr).parsedEvents.states.punish_allowcorrection,1) > 1
%     alldata(itr).parsedEvents.states
    alldata(itr).parsedEvents.states.punish_allowcorrection
    alldata(itr).parsedEvents.states.errorlick_again_wait
    alldata(itr).parsedEvents.states.errorlick_again
    alldata(itr).parsedEvents.states.punish_allowcorrection
    alldata(itr).parsedEvents.states.correctlick_again_wait
    alldata(itr).parsedEvents.states.correctlick_again
    alldata(itr).parsedEvents.states.punish_allowcorrection_done
    alldata(itr).parsedEvents.states.reward
end
%}


%% Deal with trials that mouse made an uncommitted lick first and then switched to the other side.

% trials that mouse licked the error side during the decision time. Afterwards, the mouse may have committed it
% (hence entered allow correction or punish) or may have licked the correct side.
a = arrayfun(@(x)x.parsedEvents.states.errorlick_again_wait, alldata, 'uniformoutput', 0);
errorlick_again_wait_entered = (~cellfun(@isempty, a));

% trials that mouse licked the correct side during the decision time. Afterwards, the mouse may have committed it
% (hence reward) or may have licked the error side.
a = arrayfun(@(x)x.parsedEvents.states.correctlick_again_wait, alldata, 'uniformoutput', 0);
correctlick_again_wait_entered = (~cellfun(@isempty, a));


errThenCorr = (errorlick_again_wait_entered & ~allowCorrectEntered & outcomes==1); % uncommitted error lick then committed correct lick.
corrThenErr = (correctlick_again_wait_entered & outcomes==0); % uncommitted correct lick then committed error lick.

fract_corrBut1stErr_ErrBut1stCorr = [sum(errThenCorr) / sum(outcomes==1) ,  sum(corrThenErr) / sum(outcomes==0)];
fprintf('%.2f= fract correct outcome preceded by error try\n%.2f= fract incorrect outcome preceded by correct try\n', fract_corrBut1stErr_ErrBut1stCorr)


% Some numbers to quantify animal's changes of mind:
a = arrayfun(@(x)x.parsedEvents.states.errorlick_again, alldata, 'uniformoutput', 0);
errorlick_again_entered = (~cellfun(@isempty, a));

a = arrayfun(@(x)x.parsedEvents.states.correctlick_again, alldata, 'uniformoutput', 0);
correctlick_again_entered = (~cellfun(@isempty, a));

% In what fraction of correctlick_wait trials (ie trials that animal
% made a correct lick during wait4decision), animal made incorrect lick
% afterwards?
a = nanmean(errorlick_again_entered(correctlick_again_wait_entered));
fprintf('%.2f = Fract correct try followed by error try\n', a)

% In what fraction of errorlick_wait trials (ie trials that animal
% made an error lick during wait4decision), animal made correct lick
% afterwards?
a = nanmean(correctlick_again_entered(errorlick_again_wait_entered));
fprintf('%.2f = Fract error try followed by correct try\n', a)



switch uncommittedResp
    case 'remove'  % remove them from analysis!
        allResp(errThenCorr) = NaN;
        allResp(corrThenErr) = NaN;
        
        outcomes(errThenCorr) = NaN;
        outcomes(corrThenErr) = NaN;
        
        disp('Removing correct trials that were preceded by an errorlick try.')
        disp('Removing incorrect trials that were preceded by a correctlick try.')
        
    case 'change'  % change their outcome and response to the original (1st) lick.
        aa = allResp(errThenCorr);
        a = aa;
        a(aa==1) = 2;
        a(aa==2) = 1;
        allResp(errThenCorr) = a;
        
        aa = allResp(corrThenErr);
        a = aa;
        a(aa==1) = 2;
        a(aa==2) = 1;
        allResp(corrThenErr) = a;
        
        
        outcomes(errThenCorr) = 0;
        outcomes(corrThenErr) = 1;
        
        disp('Changing the outcome and response side of correct (incorrect) trials preceded by a lick to the other side.')
        
%     case 'nothing'
        
end


%%

allResp_HR_LR = allResp;
% allResp_HR_LR(trs2rmv) = NaN;

if strcmp(alldata(1).highRateChoicePort, 'R')
    allResp_HR_LR = allResp_HR_LR-1; % 0: left (LR) , 1: right (HR)
    
elseif strcmp(alldata(1).highRateChoicePort, 'L')
    allResp_HR_LR(allResp_HR_LR==2) = 0; % 0: right (LR) ,  1: left (HR)
end


