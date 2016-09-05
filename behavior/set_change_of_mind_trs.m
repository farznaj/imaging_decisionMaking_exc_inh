%{
% mouse entered both correct lick again wait and error lick again wait

% trials that mouse licked the error side during the decision time. Afterwards, the mouse may have committed it
% (hence entered allow correction or punish) or may have licked the correct side.
a = arrayfun(@(x)x.parsedEvents.states.errorlick_again_wait, alldata, 'uniformoutput', 0);
errorlick_again_wait_entered = (~cellfun(@isempty, a));

% trials that mouse licked the correct side during the decision time. Afterwards, the mouse may have committed it
% (hence reward) or may have licked the error side.
a = arrayfun(@(x)x.parsedEvents.states.correctlick_again_wait, alldata, 'uniformoutput', 0);
correctlick_again_wait_entered = (~cellfun(@isempty, a));

%%
% find change-of-mind trials (trials that mouse made a try and then another
% try on the other side).
a = [errorlick_again_wait_entered; correctlick_again_wait_entered];
b = sum(a);
b(trs2rmv) = nan;
trs_com = (b==2); % logical array showing change-of-mind trials.

fprintf('# change-of-mind trials= %i\n', sum(trs_com))
%}

%%
% We define change-of-mind trials as trials that mouse licked one side,
% did not commit it and then licked another side. So if mouse licked error
% side, commit it, and then licked correct side, it wont be counted as a
% change of mind.


a = ~isnan(time1stCorrectTry) & ~isnan(time1stIncorrectTry);
trs_com = a; % trials that mouse tried both sides % same as trs_com defined above

% Mouse entered both. If correct try was first, we are sure mouse
% switched the side without committing the correct try (bc he cannot
% receive reward and then switch). So if correct try was first, it always
% counts as a COM. 
% But if incorrect was first, there is a chance mouse committed it, in
% which case we wont count it as COM. To find these we do:
b = ~isnan(timeCommitIncorrResp);

% (a & b) : these are the trials that mouse did both sideTries and committed the
% incorr side.... if incorrCommit came before correctTry, we dont count it
% as COM
c = (a & b) & (timeCommitIncorrResp < time1stCorrectTry); % mouse did both sideTries and incorrCommit came before correctTry .... we don't count as COM. 
% find(c)

trs_com(c) = false; % set them to zero in trs_com logical array.

fprintf('# change-of-mind trials= %i\n', sum(trs_com))


