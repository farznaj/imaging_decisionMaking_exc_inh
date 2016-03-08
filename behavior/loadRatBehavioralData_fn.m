function [data, array_days_loaded, trials_per_day] = loadRatBehavioralData_fn (subject, day, n_days_back)
% Example:
% [data, array_days_loaded, trials_per_day] = loadRatBehavioralData_fn('am053', '24-Sep-2014', 10);

%%
if nargin < 3
    n_days_back = 1;
end

folder = fullfile('Z:','data', subject, 'behavior');

list_files = dir(folder);
list_files = list_files(3:end);
[~,b] = sort([list_files(:).datenum]);
files_ordered = list_files(b);
files = {files_ordered.name};

array_days_loaded = zeros(1, n_days_back);
data = [];
trials_per_day = [];


%%
for n = 1:n_days_back
    pattern = [subject '_' day '_[0-9][0-9][0-9][0-9].mat'];
    
    %% Search the directory to find the m file of the day you want to analyze.
    for f = 1:length(files)
        this_file = files{f};
        result = regexp(this_file, pattern, 'match');
        
        if ~isempty(result) % & ismember(day, which_days);            
            % Load the all_data file and remove the last trial
            fprintf('Found file: %s\nLoading...\n\n', result{1});
            load(fullfile(folder, result{1}))
            
            all_data = all_data(1:end-1);
            
            %% Correcting field names, if data was recorded unders a different protocol. (Cleaning-up behavioral data)            
            if ~isfield(all_data,'parsedEvents')
                clean_all_data = cleanup_behav_data(all_data);
            else
                clean_all_data = all_data;
            end
            
            %%
            data = [data clean_all_data];           
            trials_per_day = [trials_per_day length(clean_all_data)];
            array_days_loaded(n) = 1;

        end
    end
    
    %% go one day back
    day = datestr(datenum(day,'dd-mmm-yyyy')-1); % Farz
    
end


%%
if length(trials_per_day) < 1
    fprintf('No session found for this day.\n\n');
elseif length(trials_per_day) > 1
    fprintf('Found more than one file. Merged %d sessions.\n', length(trials_per_day));
    fprintf('Average of %d trials/day.\n', round(mean(trials_per_day)));
    trials_per_day
else
    fprintf('%d trials.\n\n', trials_per_day);
end


