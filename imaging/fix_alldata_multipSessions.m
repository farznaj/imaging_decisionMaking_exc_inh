% Fix early sessions that a single alldata was saved for all imaging
% sessions in a day.... Here we split alldata into n files, n=number of
% sessions.

% Set the following vars

nsess = 2;
n0 = alldata_fileNames{1}; %'fni16_25-Aug-2015_1340.mat'; % file with alldata of all sessions.
% n0 = '~/Shares/Churchland/data/fni16/behavior/fni16_21-Aug-2015_1550.mat';
n0
[f1,f2,f3] = fileparts(n0);
tim = str2num(n0(end-7:end-4));
na = cell(1,nsess);
for i=1:nsess
    na{i} = fullfile(f1,[f2(1:end-4),num2str(tim-i),f3]);
    % na = {'fni16_25-Aug-2015_1337.mat', 'fni16_25-Aug-2015_1338.mat', 'fni16_25-Aug-2015_1339.mat'}; % names of new behavioral files containing alldata. Choose names earlier than the complete alldata (the one with trials of all sessions).
end
na = na(end:-1:1)
celldisp(na)

framecountFiles = cell(1,nsess);
numfrs = cell(1,nsess);
for m = 1:nsess
    framecountFiles{m} = fullfile(tifFold, sprintf('framecounts_%03d.txt', m));
    numfrs{m} = frameCountsRead(framecountFiles{m});
end

lenFrCounts = cellfun(@length, numfrs); %[93, 46, 18]; % number of trials for each session based on the framesCount files.
% lenFrCounts = [93+1, 46, 18]; % maybe do this instead, so you dont have to do all_data(2:end) below.
lenFrCounts, sum(lenFrCounts)


%% Split trials of all_data into multiple session

load(n0, 'all_data')
length(all_data)
all_data0 = all_data;

cs = [0, cumsum(lenFrCounts)];

all_datan = cell(1, length(lenFrCounts));
for i = 1: length(lenFrCounts)
    all_datan{i} = all_data(cs(i)+1:cs(i+1));
end

for i = 1: length(lenFrCounts)
    all_data = all_datan{i};
    save(na{i}, 'all_data')
end


%% Fix trial IDs and the last trial for all_data of session 2 to end. 

for ii = 1: length(lenFrCounts)

    n = na{ii};
    load(n, 'all_data')
    
    
    % fix file names.
    for i = 1:length(all_data)
        all_data(i).filename = n;
    end
    
    
    % fix trial IDs. session 1 is already good.
    if ii > 1
        id1 = all_data(1).trialId;
        for i = 1:length(all_data)
            all_data(i).trialId = all_data(i).trialId  - id1 + 1;       
        end
    end
    
    
    % add a last empty trial
    if ii < length(lenFrCounts) % last session should be good. 
        all_data(end+1) = all_data(end);
        all_data(end).trialId = all_data(end-1).trialId + 1;
        all_data(end).parsedEvents = [];
        all_data(end).responseSideIndex = [];
        all_data(end).responseSideName = [];
        all_data(end).wrongInitiation = [];
        all_data(end).didNotLickagain = [];
        all_data(end).earlyDecision = [];
        all_data(end).didNotChoose = [];
        all_data(end).didNotSideLickAgain = [];
        all_data(end).success = [];
        all_data(end).outcome = [];
        all_data(end).exclude = [];
        all_data(end).wheelRev = [];
        all_data(end).wheelSampleInt = [];
        all_data(end).wheelPostTrialToRecord = [];
    end
    
    
    
    if ii>1
        'check this'
%         all_data = all_data(2:end);
        d = length(all_data0) - cs(ii+1);
        all_data = all_data(d+1:end);
    end
    
    
    %% save the new all_data
    
    save(n, 'all_data')
    
    
end

