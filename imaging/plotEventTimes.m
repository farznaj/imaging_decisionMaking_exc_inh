% mn: minimum y of the line that marks the events
% mx: maximum y of the line that marks the events
if ~exist('mn','var')
    mn = -.1;
end
if ~exist('mx','var')
    mx = .1;
end

% mn = min(toplot(:));
% mx = max(toplot(:));

% timeInitTone, timeStimOnset, timeStimOffset, timeCommitCL_CR_Gotone, time1stSideTry, timeReward, timeCommitIncorrResp, timeStop, centerLicks, leftLicks, rightLicks

% timeInitTone
plot([timeInitTone{itr} timeInitTone{itr}]',[mn mx],'k--', 'handlevisibility', 'off')

% time1stCenterLick and timeStimOnset happen very closely relative to
% each other. (although you are compuing timeStimOnset only for valid
% trials).
%     plot([time1stCenterLick(itr) time1stCenterLick(itr)],[mn mx],'--')
plot([timeStimOnset(itr) timeStimOnset(itr)],[mn mx],'r--', 'handlevisibility', 'off')
plot([timeStimOffset(itr) timeStimOffset(itr)],[mn mx],'r--', 'handlevisibility', 'off')

% go tone, center reward
plot([timeCommitCL_CR_Gotone(itr) timeCommitCL_CR_Gotone(itr)],[mn mx],'b--', 'handlevisibility', 'off')

% 1st side try
plot([time1stSideTry(itr) time1stSideTry(itr)],[mn mx],'--', 'color', [1 .8 .1], 'handlevisibility', 'off')

% outcome
plot([timeReward(itr) timeReward(itr)],[mn mx],'m--', 'handlevisibility', 'off')
plot([timeCommitIncorrResp(itr) timeCommitIncorrResp(itr)],[mn mx],'g--', 'handlevisibility', 'off')

% stop_rotary_scope
plot([timeStop(itr) timeStop(itr)], [mn mn+(mx-mn)/2],'k--', 'handlevisibility', 'off')

% licks
lhn = NaN(1,3);
lall = {'C','L','R'};
if ~isempty(centerLicks{itr})
    lhn(1) = plot(centerLicks{itr}, (mn-.01)*ones(1, length(centerLicks{itr})),'k.', 'markersize', 14);
end

if ~isempty(leftLicks{itr})
    lhn(2) = plot(leftLicks{itr}, (mn-.01)*ones(1, length(leftLicks{itr})),'r.', 'markersize', 14);
end

if ~isempty(rightLicks{itr})
    lhn(3) = plot(rightLicks{itr}, (mn-.01)*ones(1, length(rightLicks{itr})),'g.', 'markersize', 14);
end
legend(lhn(~isnan(lhn)), lall(~isnan(lhn)))

title(['initTone  ', '{\color{red}1stCLick\_stimOn  stimOff}  ', '{\color{blue}commitCLick\_CReward\_GoTone}  ', '{\color[rgb]{1 .8 .1}1stSideTry}  ', '{\color{magenta}reward}  ', '{\color{green}commitIncorr}  ', 'stop'])


