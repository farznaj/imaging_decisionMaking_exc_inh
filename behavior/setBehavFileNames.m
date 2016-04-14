function [alldata_fileNames, days_all] = setBehavFileNames(subject, days, dayLast, days2exclude, GE)
%
% alldata_fileNames includes the name of behavioral data .mat files specified
% by the inputs:
% subject: string array
% days, dayLast, days2exclude: cell array
%
% days_all includes the date of each mat file.
%
% Example:
% Analyze all days between 02-Nov and 20-Sep, excluding 29-Sep and 12-Oct.
% days2exclude = {'12-Oct-2015', '29-Sep-2015'};
% [alldata_fileNames, days_all] = setBehavFileNames('fni17', {'02-Nov-2015'}, {'20-Sep-2015'}, days2exclude);
%
% Another usage:
% Specify what days you want to analyse, 02-Nov and 24-Sep. In this usage
% you need only 2 inputs to the function.
% [alldata_fileNames, days_all] = setBehavFileNames('fni17', {'02-Nov-2015', '24-Sep-2015'});


if ~exist('GE', 'var') % for Gamal
    GE = false;
end

%%
day = days{1};
if nargin < 3
    if length(days)==1
        n_days_back = 0;        
    else
        n_days_back = length(days)-1;
    end
else
    n_days_back = datenum(days{1},'dd-mmm-yyyy') - datenum(dayLast{1},'dd-mmm-yyyy');
end

if nargin < 4
    days2exclude = {};
end

if ~GE % Farzaneh
    if ismac
    %     dataPath = '/Users/Farzaneh/Desktop/Farzaneh/data'; % macbook
        dataPath = '/Volumes/churchland/data';
    elseif isunix
        dataPath = '/sonas-hs/churchland/nlsas/data/data'; % server
    elseif ispc
        dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % lab PC
    %     dataPath = '\\sonas-hs.cshl.edu\churchland-norepl\data not likely to be used';
    end
else % Gamal
    dataPath = '/Users/gamalamin/git_local_repository/Farzaneh/data';
end


folder = fullfile(dataPath, subject, 'behavior'); % folder = fullfile('Z:','data', subject, 'behavior');

% delete files that start with '._'
delete_dot_underline_files(subject)

list_files = dir(folder);
list_files = list_files(3:end);
[~,b] = sort([list_files(:).datenum], 'descend');
files_ordered = list_files(b);
files = {files_ordered.name};
% showcell(files')

alldata_fileNames = {};
days_all = {};
% regexp(alldata_fileNames, ['[0-9][0-9]-[A-Z][a-z][a-z]-[0-9][0-9][0-9][0-9]'])


%%
for n = 1 : (1+n_days_back)
    
    if length(days)>1
        day = days{n};
    end
    
    pattern = [subject '_' day '_[0-9][0-9][0-9][0-9].mat'];
    
    
    %% Search the directory to find the .mat file of the day you want to analyze.
    for f = 1:length(files)
        
        this_file = files{f};
        result = regexp(this_file, pattern, 'match', 'ignorecase');
        
        if ~isempty(result) % & ismember(day, which_days);            
            alldata_fileNames{end+1} = fullfile(folder, result{1});
            days_all{end+1} = datestr(datenum(day,'dd-mmm-yyyy'));
        end
    end
    
    
    %% go one day back
    if length(days)==1
        day = datestr(datenum(day,'dd-mmm-yyyy')-1); % Farz
    end
    
    
end
% showcell(alldata_fileNames')


%% remove some days if desired
for i = 1:length(days2exclude)
    alldata_fileNames(strcmp(days_all, days2exclude{i})) = [];
    days_all(strcmp(days_all, days2exclude{i})) = [];
end


%% sort files in the same day based on time ... needs work.
%{
[~,fn] = fileparts(alldata_fileNames{1});
a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
[~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
alldata_fileNames = alldata_fileNames(isf);
days_all = days_all(isf);
%}
