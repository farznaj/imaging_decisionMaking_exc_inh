function delete_dot_underline_files(subject)
% delete_dot_underline_files(subject)
%
% delete files that start with '._'
%
% subject = 'fni11';


%%
if ismac
    dataPath = '/Users/Farzaneh/Desktop/Farzaneh/data'; % macbook
elseif isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data'; % server
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data'; % lab PC
end

folder = fullfile(dataPath, subject, 'behavior');


%% find ._ files
list_files = dir(folder);
files = {list_files.name};
idot = cellfun(@(x)strfind(x,'._'), files, 'uniformoutput', 0);
idot = cellfun(@(x)~isempty(x), idot);


%% delete ._ files
if any(idot)
    showcell([files(idot)]')
    cd(folder)
    delete(files{idot})
end


