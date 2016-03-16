% mn: minimum y of the line that marks the events
% mx: maximum y of the line that marks the events
if ~exist('mn','var')
    mn = -.5;
end
if ~exist('mx','var')
    mx = .5;
end

% mn = min(toplot(:));
% mx = max(toplot(:));

% timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, timeStop, centerLicks, leftLicks, rightLicks

% timeInitTone
plot([timeInitTone{itr} timeInitTone{itr}]',[mn mx],'k--')

% time1stCenterLick and timeStimOnset happen very closely relative to
% each other. (although you are compuing timeStimOnset only for valid
% trials).
%     plot([time1stCenterLick(itr) time1stCenterLick(itr)],[mn mx],'--')
plot([timeStimOnset(itr) timeStimOnset(itr)],[mn mx],'r--')
plot([timeStimOffset(itr) timeStimOffset(itr)],[mn mx],'r--')

% go tone, center reward
plot([timeCommitCL_CR_Gotone(itr) timeCommitCL_CR_Gotone(itr)],[mn mx],'b--')

% 1st side try
plot([time1stSideTry(itr) time1stSideTry(itr)],[mn mx],'--', 'color', [1 .8 .1])

% outcome
plot([timeReward(itr) timeReward(itr)],[mn mx],'m--')
plot([timeCommitIncorrResp(itr) timeCommitIncorrResp(itr)],[mn mx],'g--')

% stop_rotary_scope
plot([timeStop(itr) timeStop(itr)], [mn mn+(mx-mn)/2],'k--')

% licks
if ~isempty(centerLicks{itr})
    plot(centerLicks{itr}, mn-.01,'k.', 'markersize', 14)
end

if ~isempty(leftLicks{itr})
    plot(leftLicks{itr}, mn-.01,'r.', 'markersize', 14)
end

if ~isempty(rightLicks{itr})
    plot(rightLicks{itr}, mn-.01,'g.', 'markersize', 14)
end

title(['initTone  ', '{\color{red}1stCLick\_stimOn  stimOff}  ', '{\color{blue}commitCLick\_CReward\_GoTone}  ', '{\color[rgb]{1 .8 .1}1stSideTry}  ', '{\color{magenta}reward}  ', '{\color{green}commitIncorr}  ', 'stop'])
