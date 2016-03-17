function [data, trials_per_session] = loadBehavData(alldata_fileNames, defaultHelpedTrs, saveHelpedTrs, doclean)
%
% [data, trials_per_session] = loadBehavData(alldata_fileNames);

if ~exist('doclean', 'var') % if true, helpedTrs and inconsistent field names will be taken care of.
    doclean = false;
end


%%
% defaultHelpedTrs = true; false;
% saveHelpedTrs = true;

% array_days_loaded = zeros(1, n_days_back);
trials_per_session = zeros(1, length(alldata_fileNames));
data = [];


%%
for f = 1:length(alldata_fileNames)
    
    %% Load all_data and remove the last trial
    
    [~,fn] = fileparts(alldata_fileNames{f});
    fprintf('Loading file: %s\n', fn);
    
    load(alldata_fileNames{f})
    
    
    if doclean
        %%  Take care of helped trials

        % show file names 
        a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))'; 
        sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0))

        all_data = setHelpedTrs(all_data, defaultHelpedTrs, saveHelpedTrs, alldata_fileNames{f});


        %% Take care of alldata fields that might be absent. (eg waitDurWarmup, adaptiveDurs, and startScopeDur) 

        all_data = cleanAlldataFields(all_data, alldata_fileNames{f});
        
    end
    
    
    %% remove the last trial
    
    all_data = all_data(1:end-1);
    
    
    %% concatenate all_data of all the mat files in alldata_fileNames
    
    data = [data  all_data];
    trials_per_session(f) = length(all_data);
%     array_days_loaded(n) = 1;
    
    
end


%%
if length(trials_per_session) < 1
    fprintf('No session found for this day.\n\n');
    
elseif length(trials_per_session) > 1
    fprintf('Merged %d sessions.\n', length(trials_per_session));
    fprintf('On average: %d trials/day.\n', round(mean(trials_per_session)));    
    fprintf('Number of trials: '), fprintf([repmat('%d  ', 1,10), '\n'], trials_per_session), fprintf('\n')
    
end


