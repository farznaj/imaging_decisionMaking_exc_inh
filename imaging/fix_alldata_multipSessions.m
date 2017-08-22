% Fix early sessions that a single alldata was saved for all imaging
% sessions in a day.... Here we split alldata into n files, n=number of
% sessions.
%{
mouse = 'fni17';
imagingFolder = '150814';
%}

% set this:
nsess = 2; %length(alldata_fileNames);


%% Set filenames

[alldata_fileNames, ~] = setBehavFileNames(mouse, {datestr(datenum(imagingFolder, 'yymmdd'))});
% sort it
[~,fn] = fileparts(alldata_fileNames{1});
a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
[~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
alldata_fileNames = alldata_fileNames(isf);

celldisp(alldata_fileNames)

[~, ~, tifFold] = setImagingAnalysisNames(mouse, imagingFolder, 1);


%% Set the following vars

n0 = alldata_fileNames{end}; %{1} %'fni16_25-Aug-2015_1340.mat'; % file with alldata of all sessions.
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

load(n0, 'all_data'), length(all_data)
all_data0 = all_data;

csImaging = [0, cumsum(lenFrCounts)];
csImaging(end) = length(all_data0);

all_datan = cell(1, length(lenFrCounts));
for i = 1: length(lenFrCounts)
    all_datan{i} = all_data(csImaging(i)+1:csImaging(i+1));
end

%%% Save the splitted all_data
%{
for i = 1: length(lenFrCounts)
    all_data = all_datan{i};
    save(na{i}, 'all_data')
end
%}

%% Fix trial IDs and the last trial for all_data of session 2 to end. 

all_datan2 = cell(1, length(lenFrCounts));

for ii = 1: length(lenFrCounts) % ii=2;

    n = na{ii};
    all_data = all_datan{ii};
%     load(n, 'all_data')
    
    
    % fix file names.
    for i = 1:length(all_data)
        all_data(i).filename = n;
    end
    
    % Figure out what trial of all_data0 corresponds to first trial in the 2nd frameCount file.      
    if ii>1  && ii==length(lenFrCounts)% some of the first few trials of session 2 may not be in imaging, so remove them.

        % Compute number of frames from all_data0

        frameLength = 1000 / 30.9;
        nfrs = NaN(1, length(all_data)-1); % min(length(all_data)-1, length(framesPerTrial)));
        for tr = 1 : length(nfrs) % min(length(all_data)-1, length(framesPerTrial))
            % duration of a trial in mscan (ie duration of scopeTTL being sent).
            durtr = all_data(tr).parsedEvents.states.stop_rotary_scope(1)*1000 + 500 - ...
                all_data(tr).parsedEvents.states.start_rotary_scope(1)*1000; % 500 is added bc the duration of stop_rotary_scope is 500ms.
            nfrs(tr) = durtr/ frameLength;
        end


        %%% Set number of frames from the frameCount file
%         frameCountFileName = 'framecounts_002.txt'; % params.framecountFiles{2}
%         nfrs_mscan = frameCountsRead(frameCountFileName);
        nfrs_mscan = numfrs{ii};

        %%%
        mn = min(length(nfrs), length(nfrs_mscan));
        figure; hold on;
        plot(nfrs); %(1:mn)); 
        plot(nfrs_mscan)


        %%% Figure out what trial of all_data0 matches the 1st trial of frameCount file in terms of number of frames
        mn = min(length(nfrs), length(nfrs_mscan));

        jall = [];
        tr = 20; 
        for j = 1:30 %lenFrCounts(1)+30 %1-tr : 30
            b = abs(nfrs_mscan(tr) - nfrs(tr+j))<2; 
            if b==1 
                jall = [jall, j]; 
            end
        end
        disp(jall)
        d = jall(1);  % the first one is perhaps the amount of shift you need to apply.
        d = jall(end);

        %%%
        nfrs_alldata = nfrs(1+d : min(length(nfrs), d+mn));
        plot(nfrs(1+d : min(length(nfrs), 1+d+mn))); % now plot the alldata0-computed frames from the trial that corresponds to 1st trial of frameCount file
        % it should now match nfrs_mscan
        mn2 = min(length(nfrs_alldata), length(nfrs_mscan));
        figure; plot(nfrs_alldata(1:mn2) -nfrs_mscan(1:mn2)) 
        if max(abs((nfrs_alldata(1:mn2) -nfrs_mscan(1:mn2)))) > 2.5
            error('something wrong')
        end
        
        %%%        
        all_data = all_data(d+1:end);
        cprintf('m', 'starting from trial %d of all_data0...\n', d+1)
        
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

    
    
    %% save the new all_data
    
    all_datan2{ii} = all_data;
%     save(n, 'all_data')
    
end


%% Save the new all_data files

for ii = 1: length(lenFrCounts) % ii=2;    
    all_data = all_datan2{ii};
    save(na{ii}, 'all_data')
end



%%
%         warning('check this')
%         all_data = all_data(2:end);
        %{
        d = length(all_data0) - csImaging(ii+1);
        cprintf('m', 'starting from trial %d of all_data...\n', d+1)
        all_data = all_data(d+1:end);        
        %{
        % check for a random trial (tr)
        tr = 7; for j=1:20, b = isequaln(all_datan{ii}(tr).parsedEvents.states, all_data0(tr+j).parsedEvents.states); if b==1, disp(j), break, end, end
        if j~=d+1
            error('something wrong; the 2 methods dont match!')
        end
        %}
        %}
        %{
        % one guess is that the last trial in frameCount file matches the last
        % trial in all_data0... we go with this guess and later when we do
        % trialization we can confirm it or come back here and change it.
        d = length(all_data0) - lenFrCounts(ii);
        all_data = all_data0(d+1:end);
        cprintf('m', 'starting from trial %d of all_data0...\n', d+1)
        %}

