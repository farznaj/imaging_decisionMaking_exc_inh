function extraTrs = setRandExtraTrs(cond1, cond2)
% if LR and HR have different number of trials, get a list of random trials
% from the conditon with more trials, so you can later remove those extra
% trials.

% lr = find(contraTrs);
% hr = find(ipsiTrs);

% lr = find(choiceVec==0);
% hr = find(choiceVec==1);

dl = abs(length(cond1) - length(cond2)); % difference between number of trials between the 2 conditions.

if dl~=0
    if length(cond1) > length(cond2)
        extraTrs = randperm(length(cond1)); % identify dl random indeces from the condition with more trials.
        extraTrs = extraTrs(1:dl);     % extraTrs = randi(length(lr), 1, dl);  % it may have repetitions so we dont use it.
        extraTrs = cond1(extraTrs); % indeces of choiceVec that u should take out so both lr and hr have the same number of trials.

    elseif length(cond2) > length(cond1)
        extraTrs = randperm(length(cond2)); % identify dl random indeces from the condition with more trials.
        extraTrs = extraTrs(1:dl);    
        extraTrs = cond2(extraTrs);
    end
else
    extraTrs = [];
end


%%
%{
fprintf('N trials for LR and HR = %d  %d\n', [sum(choiceVec==0), sum(choiceVec==1)])

% make sure choiceVec has equal number of trials for both lr and hr.
choiceVec(extraTrs) = NaN; % set to nan some trials (randomly chosen) of the condition with more trials so both conditions have the same number of trials.

contraTrs(extraTrs) = 0;
ipsiTrs(extraTrs) = 0;
%}

