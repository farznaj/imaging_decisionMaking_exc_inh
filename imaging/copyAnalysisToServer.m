mousename = 'FNI19'; fprintf(['\n====================',mousename,'====================\n'])
% day = '150904';
cd(fullfile(dir0, mousename))
a = dir;
days = {a.name};

for i = 3:length(days)
    day = days{i};
    
    anadir_server = fullfile('\\sonas-hs.cshl.edu\churchland\data', mousename, 'analysis', day);
    dir0 = 'C:\Users\fnajafi\Documents\Data\';
    anadir = fullfile(dir0, mousename, day, 'analysis');
    % dir01 = fullfile(dir0, mousename, day);
    
    if ~exist(anadir_server, 'dir'), mkdir(anadir_server), end
    
    copyIfNotCopied(anadir, anadir_server)
    
end



