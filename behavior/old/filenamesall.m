ratname = 'am053';
daysall = {'20-Aug' , '25-Aug', '26-Aug', '27-Aug'}; % '27-Aug-2014';
% day = '27-Aug';

% n_days_back = 2;

folder = fullfile('Z:','data', ratname, 'behavior');

list_files = dir(folder);
list_files = list_files(3:end);
[junk,b] = sort([list_files(:).datenum]);
files_ordered = list_files(b);
files = {files_ordered.name};


for n = 1:length(daysall)
%     pattern = [ratname '_' day '_[0-9][0-9][0-9][0-9].mat'];
    pattern = [ratname '_' daysall{n}]; 
%     flag_day_loaded = 0;
    for f = 1:length(files)
        this_file = files{f};
        result = regexp(this_file, pattern, 'match');
        if ~isempty(result) % & ismember(day, which_days);
%             fprintf('Found file: %s\nLoading...\n\n', result{1});
%             load([folder result{1}]);

            filename_all{n} = fullfile(folder, this_file);
%             load(fullfile(folder, result{1}))

%             disp('______________')
%             pause
        end
    end
end


%%