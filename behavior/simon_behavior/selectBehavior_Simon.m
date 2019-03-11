function bhv = selectBehavior(bhv,sessions)
% Function to select a subset of trials/settings from selected 'sessions' in a 
% larger array 'bhv' that has behavioral data. 'sessions' should be a vector of 
% session numbers that can be used and 'bhv' should include a field 
% 'SessionNr' that is used to indicate session identity. If this field is
% missing, only session data can be selected (no individual trial data).
% Usage: bhv = selectBehavior(bhv,sessions)

%% get fieldnames
if isempty(bhv)
    bFields = {};
else
    bFields = fieldnames(bhv);
end
if isfield(bhv,'SessionNr')
    trials = ismember(bhv.SessionNr,sessions); %identiy trials that correspond to selected sessions
else
    trials = []; %don't collect individual trial
end

%% cycle trough fields and carry over selected trials / sessions
for iFields = 1:size(bFields,1)
    if length(bhv.(bFields{iFields})) ~= length(trials) %if field does not contain single trials, it should contain session data instead
        try
            bhv.(bFields{iFields}) = bhv.(bFields{iFields})(:,sessions); %carry over selected sessions
        catch
            bhv = rmfield(bhv,bFields{iFields}); %field does not contain trials and cant be selected with session index... remove from array.
        end
    else
        if any(size(bhv.(bFields{iFields})) == 1)
            bhv.(bFields{iFields}) = bhv.(bFields{iFields})(trials); %carry over selected trials
        else
            bhv.(bFields{iFields}) = bhv.(bFields{iFields})(:,trials); %carry over selected trials
        end
    end 
end
