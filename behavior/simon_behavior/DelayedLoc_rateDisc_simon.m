function [Performance,bhv] = DelayedLoc_rateDisc_simon(Animal,cPath,lSessions,binSize,highDetection,showPlot,minDelay)
% Analyze behavioral data from rate discrimination task to test for the
% impact of optogenetic manipulation.
%
% Optional inputs:
% lSessions: Only use last 'lSessions' sessions for behavioral analysis.
% binSize: Number of bins used to compute discrimination. Default is 10.
% highDetection: Only use sessions at which detection was at 90% or higher.
% minDelay: Only use sessions at which at least one decisionGap > 'minDelay' was presented. The default is all trials.
% showPlot: Plot behavioral results. Default is true.

%% check optional input
if ~exist('lSessions','var') || isempty(lSessions)
    lSessions = inf;
end

if ~exist('highDetection','var') || isempty(highDetection)
    highDetection = 0;
end

if ~exist('showPlot','var') || isempty(showPlot)
    showPlot = true;
end

if ~exist('minDelay','var') || isempty(minDelay)
    minDelay = 0;
end

if ~exist('binSize','var') || isempty(binSize)
    binSize = 10;
end

if ~strcmpi(cPath(end),filesep)
    cPath = [cPath filesep];
end
modId = [2 4 6]; %id for modalities. Default is audio, somatosensory and audiosomatosensory
modLabels = {'audio' 'tactile' 'audiotactile'}; %labels for different modalities
modColor = ['r' 'b' 'k']; %colors for different modalities
nrBlocks = 5; %number of blocks the data is separated in later

cPath0 = cPath;

%% assign some variables

paradigm = 'SpatialDisc';
for iChecks = 1:10 %check for files repeatedly. Sometimes the server needs a moment to be indexed correctly
    Files = dir([cPath0 '/' Animal '/' paradigm '/Session Data/*' Animal '*']); %behavioral files in correct cPath
    if ~isempty(Files)
        break;
    end
    pause(0.1);
end

Files = cell2mat({Files.name}');

cPath = [cPath Animal '/' paradigm '/Session Data/']; %folder with behavioral data
maxTrialCnt = 1000; %maximum trials per datapoint
maxDelay = 3; %maximum delay in seconds

cDate = datenum(Files(:,length([Animal '_' paradigm '_'])+1:length([Animal '_' paradigm '_'])+10)); %isolate dates from Filenames
cDate = cDate + (str2num(Files(:,length([Animal '_' paradigm '_'])+19:end-4))*0.01); %add session nr to timestamp

[cDate,ind] = sort(cDate,'descend'); %sort counts to get the order of files to days correct. Newest file should be first in the list.
cDate = floor(cDate);
Files = Files(ind,:); %adjust order of filenames to get it to be chronological

%%
% exclude
pattern = 'Feb01_2019_Session3';
for i=1:size(Files,1)

    a = strfind(Files(i,:), pattern);
    if ~isempty(a)
        break
    end
end
i3 = i;

Files(i3,:) = [];



% start of TST (tail suspension test)
pattern = 'Jan30_2019_Session1';
for i=1:size(Files,1)

    a = strfind(Files(i,:), pattern);
    if ~isempty(a)
        break
    end
end
i1 = i;


% end of TST (tail suspension test)
pattern = 'Feb21_2019_Session2';
for i=1:size(Files,1)

    a = strfind(Files(i,:), pattern);
    if ~isempty(a)
        break
    end
end
i2 = i;



% % exclude
% pattern = 'Feb01_2019_Session3';
% for i=1:size(Files,1)
% 
%     a = strfind(Files(i,:), pattern);
%     if ~isempty(a)
%         break
%     end
% end
% i3 = i;
% 
% a = 1:i1;
% a(i3) = [];

% get only sessions with stress paradigm
a = i2:i1;

% Files(a,:)


%% session 1 data (TST sessions, before the stress)

inds = a(2:2:end);
sess1 = Files(inds,:);
sess2plot = sess1;


%% session 2 data (TST sessions, after the stress)

inds = a(1:2:end);
sess2 = Files(inds,:);
sess2plot = sess2;


%% all sessions before TST sessions

inds = i1+2 : size(Files,1);
inds = i1+2 : i1+2+4; % only the last 5 sessions before TST sessions (because in the very early sessions, the mouse is not expert).
sess2plot = Files(inds,:);


%% all sessions after TST sessions

inds = 1: i2-1;
sess2plot = Files(inds,:);


%%
disp(sess2plot)


%% load data
bhv = []; Cnt = 0;

for iFiles = 1:size(sess2plot,1)
    
    iFilesnow = sess2plot(iFiles,:);
    
    load([cPath iFilesnow], 'SessionData'); %load current bhv file
    if isfield(SessionData,'Rewarded')
        SessionData.Rewarded = logical(SessionData.Rewarded);
    end
    
    useData = length(SessionData.Rewarded) > 100 && length(unique(SessionData.DistStim)) > 1 && sum(SessionData.DistStim > 0) > 50; % if file contains at least 100 trials and different distractor rates
    
    if minDelay > 0
        useData = isfield(SessionData,'decisionGap') && length(SessionData.Rewarded) > 100 && any(SessionData.decisionGap > minDelay); % if file contains at least 100 trials and sufficiently long waiting periods
    end
    
    
    if useData
        Cnt = Cnt+1;
%         DayNr(Cnt) = cDate(iFiles); % should be: 
%         for session 2:  aa =
%         a(1:2:end); aa(iFiles); for session 1:  aa = a(2:2:end); aa(iFiles)
        Performance.fileName{Cnt} = iFilesnow;
        
        for iMod = 1:3
            %% get some single session performance data
            ind = ~SessionData.DidNotChoose & ~SessionData.DidNotLever & logical(SessionData.Assisted) & SessionData.StimType == modId(iMod); %only use active trials
            Performance.SelfPerformed(iMod,Cnt) = sum(SessionData.Rewarded(ind))/sum(SessionData.Rewarded(ind)+SessionData.Punished(ind)); %peformance for self-performed trials
            
            dInd = logical(SessionData.Assisted & SessionData.DistStim == 0); %index for trials that were detection only
            Performance.Detection(iMod,Cnt) = sum(SessionData.Rewarded(ind & dInd))/sum(SessionData.Rewarded(ind & dInd)+SessionData.Punished(ind & dInd)); %peformance for detection trials
            Performance.Discrimination(iMod,Cnt) = sum(SessionData.Rewarded(ind & ~dInd))/sum(SessionData.Rewarded(ind & ~dInd)+SessionData.Punished(ind & ~dInd)); %peformance for discrimination trials
            
            lInd = SessionData.CorrectSide == 1; %index for left-choice trials
            Performance.LeftPerformed(iMod,Cnt) = sum(SessionData.Rewarded(ind & lInd))/sum(SessionData.Rewarded(ind & lInd)+SessionData.Punished(ind & lInd));
            Performance.RightPerformed(iMod,Cnt) = sum(SessionData.Rewarded(ind & ~lInd))/sum(SessionData.Rewarded(ind & ~lInd)+SessionData.Punished(ind & ~lInd));
            
%             Performance.Date{1,Cnt} = datestr(cDate(Cnt)); % FN: this needs to change
            Performance.Date{1,Cnt} = datestr(cDate(inds(Cnt))); 
            
            
            %% compute discrimination performance and stats
            
            if ~isnan(Performance.Discrimination(iMod,Cnt))
                rInd = (SessionData.CorrectSide == 1 & SessionData.Punished) | (SessionData.CorrectSide == 2 & SessionData.Rewarded); %right-choice trials
                rInd = rInd(ind);
                
                tCnt = 0;
                eventCnt = zeros(2,sum(ind));
                for iTrials = find(ind)
                    tCnt = tCnt +1;
                    [left, right] = Behavior_getStimEvent_simon(SessionData.StimType(iTrials), SessionData.stimEvents{iTrials});
                    eventCnt(1,tCnt) = length(right);
                    eventCnt(2,tCnt) = length(left) + length(right);
                end
                
                [nTrials, distRatio, trialInd]  = histcounts(eventCnt(1,:) ./ eventCnt(2,:), binSize); %get ratio between rightward and absolute number of events
                distRatio = distRatio + diff(distRatio(1:2))/2; distRatio(end) = [];
                for iBins = 1:length(nTrials)
                    rightChoice(iBins) = sum(rInd(trialInd == iBins)); %get number of rightward trials for each difficulty
                end
                
                [params, ~, ~, cFit] = Behavior_fitPalamedes_simon(distRatio, rightChoice, nTrials); %complete discrimination parameters
                [params1, ~, ~, cFit1] = Behavior_fitPalamedes_simon(distRatio(2:end-1), rightChoice(2:end-1), nTrials(2:end-1)); %complete discrimination parameters
                Performance.cFit{iMod, Cnt} = cFit;
                Performance.disc_Fit{iMod, Cnt} = cFit1;
                fNames = fieldnames(params);
                for iFields = 1 : length(fNames)
                    Performance.(fNames{iFields})(iMod, Cnt) = params.(fNames{iFields});
                    Performance.(['disc_' fNames{iFields}])(iMod, Cnt) = params1.(fNames{iFields});
                end
            end
        end
        %% combine into one larger array
        SessionData.SessionNr = repmat(Cnt,1,SessionData.nTrials); %tag all trials in current dataset with session nr
        bhv = appendBehavior_Simon(bhv,SessionData); %append into larger array
        
        if Cnt >= lSessions
            break;
        end
    end
end
disp(['Current subject: ' Animal '; Using ' num2str(Cnt) '/' num2str(size(sess2plot,1)) ' files']);



%%
if Cnt > 0
    
    %% check for last and high performance sessions
    
    sessionSelect = 1:Cnt; %all sessions
%     sessionSelect = a(1:2:end);
    
    if highDetection > 0
        lowInd = Performance.Detection < highDetection; %find sessions with low detection performance
    else
        lowInd = false(1,Cnt);
    end
    sessionSelect(lowInd) = []; %don't use low performance sessions of selected
    disp(['Rejected ' num2str(sum(lowInd)) '/' num2str(length(lowInd))  ' files for detection performance below ' num2str(highDetection*100) '%.']);
    
    %only use last 'lSessions' days
    bhv = selectBehavior_Simon(bhv,sessionSelect); %only use trials from selecteded sessions
    Performance = selectBehavior_Simon(Performance,sessionSelect); %only use performance from selecteded sessions
    
%     DayNr = DayNr(sessionSelect) - DayNr(end) + 1; %convert Days into relative values, starting at 1 for first training day
%     SessionNr = length(DayNr):-1:1;
    fDay = Performance.Date{end}; %first date in dataset
    lDay = Performance.Date{1}; %last date in dataset
    disp(['First date: ' fDay]);
    disp(['Last date: ' lDay]);
    
    
    %% compute performance for all data combined
    
    for iMod = 1 : 3
        ind = ~bhv.DidNotChoose & ~bhv.DidNotLever & logical(bhv.Assisted) & bhv.StimType == modId(iMod); %only use active trials
        allPerf.SelfPerformed(iMod) = sum(bhv.Rewarded(ind))/sum(bhv.Rewarded(ind)+bhv.Punished(ind)); %peformance for self-performed trials
        
        dInd = logical(bhv.Assisted & bhv.DistStim == 0); %index for trials that were detection only
        allPerf.Detection(iMod) = sum(bhv.Rewarded(ind & dInd))/sum(bhv.Rewarded(ind & dInd)+bhv.Punished(ind & dInd)); %peformance for detection trials
        allPerf.Discrimination(iMod) = sum(bhv.Rewarded(ind & ~dInd))/sum(bhv.Rewarded(ind & ~dInd)+bhv.Punished(ind & ~dInd)); %peformance for discrimination trials
        
        lInd = bhv.CorrectSide == 1; %index for left-choice trials
        allPerf.LeftPerformed(iMod) = sum(bhv.Rewarded(ind & lInd))/sum(bhv.Rewarded(ind & lInd)+bhv.Punished(ind & lInd));
        allPerf.RightPerformed(iMod) = sum(bhv.Rewarded(ind & ~lInd))/sum(bhv.Rewarded(ind & ~lInd)+bhv.Punished(ind & ~lInd));
        
        %% compute discrimination performance and stats
        
        if ~isnan(allPerf.Discrimination(iMod))
            rInd = (bhv.CorrectSide == 1 & bhv.Punished) | (bhv.CorrectSide == 2 & bhv.Rewarded); %right-choice trials
            rInd = rInd(ind);
            
            tCnt = 0;
            eventCnt = zeros(2,sum(ind));
            for iTrials = find(ind)
                tCnt = tCnt +1;
                [left, right] = Behavior_getStimEvent_simon(bhv.StimType(iTrials), bhv.stimEvents{iTrials});
                eventCnt(1,tCnt) = length(right);
                eventCnt(2,tCnt) = length(left) + length(right);
            end
            
            [nTrials, distRatio, trialInd]  = histcounts(eventCnt(1,:) ./ eventCnt(2,:), binSize); %get ratio between rightward and absolute number of events
            distRatio = distRatio + diff(distRatio(1:2))/2; distRatio(end) = [];
            for iBins = 1:length(nTrials)
                rightChoice(iBins) = sum(rInd(trialInd == iBins)); %get number of rightward trials for each difficulty
            end
            [params, h1, ~, cFit] = Behavior_fitPalamedes_simon(distRatio, rightChoice, nTrials, showPlot, true); %complete discrimination parameters
            title(h1.data.Parent,[Animal ' - ' modLabels{iMod} ' - All perf. trials'])
            ylim(h1.data.Parent, [0 1]);
            allPerf.cFit{iMod} = cFit;
            
            fNames = fieldnames(params);
            for iFields = 1 : length(fNames)
                allPerf.(fNames{iFields})(iMod) = params.(fNames{iFields});
            end
            
            % discrimination only
            [params, ~, ~, cFit] = Behavior_fitPalamedes_simon(distRatio(2:end-1), rightChoice(2:end-1), nTrials(2:end-1)); %complete discrimination parameters
            allPerf.disc_cFit{iMod} = cFit;
            if showPlot
                hold(h1.fit.Parent, 'on');
                h2 = plot(h1.fit.Parent,cFit(1,:),cFit(2,:),'color','r', 'linewidth', 4);
                uistack(h2,'bottom');
                hold(h1.fit.Parent, 'off');
            end
            
            fNames = fieldnames(params);
            for iFields = 1 : length(fNames)
                allPerf.(['disc_' fNames{iFields}])(iMod) = params.(fNames{iFields});
            end
            
            %% compute performance and fits for 'nrBlocks' blocks of total data
            ind = ~bhv.DidNotChoose & ~bhv.DidNotLever & logical(bhv.Assisted) & bhv.StimType == modId(iMod); %only use active trials
            allPerf.SelfPerformed(iMod) = sum(bhv.Rewarded(ind))/sum(bhv.Rewarded(ind)+bhv.Punished(ind)); %peformance for self-performed trials
            
            dInd = logical(bhv.Assisted & bhv.DistStim == 0); %index for trials that were detection only
            allPerf.Detection(iMod) = sum(bhv.Rewarded(ind & dInd))/sum(bhv.Rewarded(ind & dInd)+bhv.Punished(ind & dInd)); %peformance for detection trials
            allPerf.Discrimination(iMod) = sum(bhv.Rewarded(ind & ~dInd))/sum(bhv.Rewarded(ind & ~dInd)+bhv.Punished(ind & ~dInd)); %peformance for discrimination trials
            
            lInd = bhv.CorrectSide == 1; %index for left-choice trials
            allPerf.LeftPerformed(iMod) = sum(bhv.Rewarded(ind & lInd))/sum(bhv.Rewarded(ind & lInd)+bhv.Punished(ind & lInd));
            allPerf.RightPerformed(iMod) = sum(bhv.Rewarded(ind & ~lInd))/sum(bhv.Rewarded(ind & ~lInd)+bhv.Punished(ind & ~lInd));
            
            clear rightChoice
            bInd = find(ind); %find indices to go through them blockwise
            bSize = 1 : floor(sum(ind) / nrBlocks) : sum(ind);
            for iRuns = 1 : length(bSize) - 1
                
                blockInd = bInd(bSize(iRuns):bSize(iRuns+1)); %current block
                blockIndDetect = blockInd(ismember(blockInd, find(ind & dInd))); %detection trials
                blockIndDisc = blockInd(~ismember(blockInd, find(ind & dInd))); %discrimination trials
                blockIndLeft = blockInd(ismember(blockInd, find(ind & lInd))); %left trials
                blockIndRight = blockInd(~ismember(blockInd, find(ind & lInd))); %right trials
                
                Blocks.SelfPerformed(iMod, iRuns) = sum(bhv.Rewarded(blockInd))/sum(bhv.Rewarded(blockInd)+bhv.Punished(blockInd)); %peformance for self-performed trials
                Blocks.Detection(iMod, iRuns) = sum(bhv.Rewarded(blockIndDetect))/sum(bhv.Rewarded(blockIndDetect)+bhv.Punished(blockIndDetect)); %peformance for detection trials
                Blocks.Discrimination(iMod, iRuns) = sum(bhv.Rewarded(blockIndDisc))/sum(bhv.Rewarded(blockIndDisc)+bhv.Punished(blockIndDisc)); %peformance for discrimination trials
                
                Blocks.LeftPerformed(iMod, iRuns) = sum(bhv.Rewarded(blockIndLeft))/sum(bhv.Rewarded(blockIndLeft)+bhv.Punished(blockIndLeft)); %peformance for leftward trials
                Blocks.RightPerformed(iMod, iRuns) = sum(bhv.Rewarded(blockIndRight))/sum(bhv.Rewarded(blockIndRight)+bhv.Punished(blockIndRight)); %peformance for rightward trials
                
                % get psychophysical curve
                [nTrials, distRatio, trialInd]  = histcounts(eventCnt(1,bSize(iRuns):bSize(iRuns+1)) ./ eventCnt(2,bSize(iRuns):bSize(iRuns+1)), binSize); %get ratio between rightward and absolute number of events
                distRatio = distRatio + diff(distRatio(1:2))/2; distRatio(end) = [];
                cInd = rInd(bSize(iRuns):bSize(iRuns+1)); %use subset of right-choice index
                for iBins = 1:length(nTrials)
                    rightChoice(iBins) = sum(cInd(trialInd == iBins)); %get number of rightward trials for each difficulty
                end
                try
                    [params, ~, ~, cFit] = Behavior_fitPalamedes(distRatio, rightChoice, nTrials, false, true); %complete discrimination parameters
                    [params1, ~, ~, cFit1] = Behavior_fitPalamedes(distRatio(2:end-1), rightChoice(2:end-1), nTrials(2:end-1)); %complete discrimination parameters
                    Blocks.cFit{iMod, iRuns} = cFit;
                    Blocks.disc_Fit{iMod, iRuns} = cFit1;
                catch
                    Blocks.cFit{iMod, iRuns} = NaN;
                    Blocks.disc_Fit{iMod, iRuns} = NaN;
                end
                    
                fNames = fieldnames(params);
                for iFields = 1 : length(fNames)
                    Blocks.(fNames{iFields})(iMod, iRuns) = params.(fNames{iFields});
                    Blocks.(['disc_' fNames{iFields}])(iMod, iRuns) = params1.(fNames{iFields});
                end
            end
        end
    end
    
    %% make plots
    
    if showPlot
        for iMod = 1 : size(Performance.cFit, 1)
            
            %% Overview figure for psychophysical fits for each modality
            
            figure('name',[Animal ' - Learning curves; Start date: ' fDay ' ; End date: ' lDay])
            h = subplot(2,2,1); hold on; gca; %plot fits for single sessions
            h.ColorOrder = imresize(colormap('gray'),[length(Performance.cFit) 3]); %ensure plotted lines are different shades of gray
            for iCurves = size(Performance.cFit, 2) : - 1 : 1
                if ~isempty(Performance.cFit{iMod, iCurves})
                    plot(Performance.cFit{iMod, iCurves}(1,:),Performance.cFit{iMod, iCurves}(2,:), 'linewidth', 2)
                end
            end
            hold off
            axis square; title([Animal ' - ' modLabels{iMod} ' session fits']);
            xlabel('Stimulus strength'); ylabel('Proportion chose high');
            ylim([0 1]);
            
            h = subplot(2,2,2); hold on; gca; %plot fits for single sessions
            h.ColorOrder = imresize(colormap('gray'),[length(Performance.disc_Fit) 3]); %ensure plotted lines are different shades of gray
            for iCurves = size(Performance.cFit, 2) : - 1 : 1
                if ~isempty(Performance.disc_Fit{iMod, iCurves})
                    plot(Performance.disc_Fit{iMod, iCurves}(1,:),Performance.disc_Fit{iMod, iCurves}(2,:), 'linewidth', 2)
                end
            end
            hold off
            axis square; title([Animal ' - ' modLabels{iMod} ' session discrimination fits']);
            xlabel('Stimulus strength'); ylabel('Proportion chose high');
            ylim([0 1]);
            
            h = subplot(2,2,3); hold on; gca;
            h.ColorOrder = imresize(colormap('gray'),[length(Blocks.cFit) 3]); %ensure plotted lines are different shades of gray
            for iCurves = 1 : size(Blocks.cFit, 2)
                try
                    plot(Blocks.cFit{iMod, iCurves}(1,:),Blocks.cFit{iMod, iCurves}(2,:), 'linewidth', 2)
                end
            end
            hold off
            axis square; title([Animal ' - ' modLabels{iMod} ' block fits']);
            xlabel('Stimulus strength'); ylabel('Proportion chose high');
            ylim([0 1]);
            
            h = subplot(2,2,4); hold on; gca;
            h.ColorOrder = imresize(colormap('gray'),[length(Blocks.cFit) 3]); %ensure plotted lines are different shades of gray
            for iCurves = 1 : size(Blocks.cFit, 2)
                try
                    plot(Blocks.disc_Fit{iMod, iCurves}(1,:),Blocks.disc_Fit{iMod, iCurves}(2,:), 'linewidth', 2)
                end
            end
            hold off
            ylim([0 1]);
            axis square; title([Animal ' - ' modLabels{iMod} ' block discrimination fits']);
            xlabel('Stimulus strength'); ylabel('Proportion chose high');
        end
        
        
        %% combined fits for all modalities
        
        if isfield(allPerf, 'cFit')
            figure
            subplot(2, 1, 1); hold on;
            for iMod = 1 : size(allPerf.cFit, 2)
                plot(allPerf.cFit{iMod}(1,:), allPerf.cFit{iMod}(2,:),'color','r', 'linewidth', 4, 'color', modColor(iMod));
                axis square
            end
            title([Animal ' - Full curve']);
            xlabel('Stimulus strength'); ylabel('Proportion chose high');
            legend(modLabels,'location', 'northwest')
            ylim([0 1]);

            subplot(2, 1, 2); hold on;
            for iMod = 1 : size(allPerf.cFit, 2)
                plot(allPerf.disc_cFit{iMod}(1,:), allPerf.disc_cFit{iMod}(2,:),'color','r', 'linewidth', 4, 'color', modColor(iMod));
                axis square
            end
            title([Animal ' - Discrimination curve']);
            xlabel('Stimulus strength'); ylabel('Proportion chose high');
            legend(modLabels,'location', 'northwest');
            ylim([0 1]);
        end
        %% cross session / blocks plots
%         figure('name',[Animal ' - Learning curves; Start date: ' fDay ' ; End date: ' lDay])
%         subplot(2,2,1); hold on
%         for iMod = 1:3
%             plot(SessionNr,Performance.Detection(iMod,:),'-o','linewidth',2, 'color', modColor(iMod));hold on
%             plot(SessionNr,Performance.Discrimination(iMod,:),'--o','linewidth',2, 'color', modColor(iMod));
%         end
%         
%         line([0 length(SessionNr)+1],[0.5 0.5],'linestyle','--','linewidth',1,'color',[0.5 0.5 0.5])
%         title([Animal ' - Performance / Sessions']); xlabel('#sessions','fontsize',15); ylabel('performance (%)','fontsize',15);
%         axis square; set(gca,'FontSize',12); ylim([0.4 1.05]);
%         xlim([0 length(Performance.sensitivity)+1])
%         
%         line([0 length(SessionNr)+1],[0.5 0.5],'linestyle','--','linewidth',1,'color',[0.5 0.5 0.5])
%         title([Animal ' - Detection Performance']); xlabel('#sessions','fontsize',15); ylabel('performance (%)','fontsize',15);
%         axis square; set(gca,'FontSize',12); ylim([0.4 1.05]);
%         
%         subplot(2,2,2); hold on;
%         for iMod = 1 : size(Performance.sensitivity, 1)
%             plot(SessionNr,Performance.sensitivity(iMod,:),'-o','linewidth',2, 'color', modColor(iMod));hold on
%             plot(SessionNr,Performance.disc_sensitivity(iMod,:),'--o','linewidth',2, 'color', modColor(iMod));
%             plot(SessionNr,Performance.bias(iMod,:),'--x','linewidth',2, 'color', modColor(iMod));
%         end
%         title([Animal ' - Sensitivity / Sessions']); xlabel('#sessions','fontsize',15); ylabel('Sensitiviy / bias','fontsize',15);
%         axis square; set(gca,'FontSize',12); ylim([0 5]);
%         xlim([0 length(Performance.sensitivity)+1])
%         line([0 length(SessionNr)+1],[0.5 0.5],'linestyle','--','linewidth',1,'color',[0.5 0.5 0.5])
%         ylim([0 10]);
%         
%         %same thing as sessions but over blocks
%         subplot(2,2,3); hold on
%         for iMod = 1 : size(Performance.sensitivity, 1)
%             plot(Blocks.Detection(iMod,:),'-o','linewidth',2, 'color', modColor(iMod));hold on
%             plot(Blocks.Discrimination(iMod,:),'--o','linewidth',2, 'color', modColor(iMod));
%         end
%         
%         line([0 length(SessionNr)+1],[0.5 0.5],'linestyle','--','linewidth',1,'color',[0.5 0.5 0.5])
%         title([Animal ' - Blocks / Sessions']); xlabel('#sessions','fontsize',15); ylabel('Blocks (%)','fontsize',15);
%         axis square; set(gca,'FontSize',12); ylim([0.4 1.05]);
%         xlim([0 length(Blocks.sensitivity)+1])
%         
%         line([0 length(SessionNr)+1],[0.5 0.5],'linestyle','--','linewidth',1,'color',[0.5 0.5 0.5])
%         title([Animal ' - Detection Blocks']); xlabel('#sessions','fontsize',15); ylabel('Blocks (%)','fontsize',15);
%         axis square; set(gca,'FontSize',12); ylim([0.4 1.05]);
%         
%         subplot(2,2,4); hold on;
%         for iMod = 1 : size(Blocks.sensitivity, 1)
%             plot(Blocks.sensitivity(iMod,:),'-o','linewidth',2, 'color', modColor(iMod));hold on
%             plot(Blocks.disc_sensitivity(iMod,:),'--o','linewidth',2, 'color', modColor(iMod));
%             plot(Blocks.bias(iMod,:),'--x','linewidth',2, 'color', modColor(iMod));
%         end
%         title([Animal ' - Sensitivity / Sessions']); xlabel('#sessions','fontsize',15); ylabel('Sensitiviy / bias','fontsize',15);
%         axis square; set(gca,'FontSize',12); ylim([0 5]);
%         xlim([0 length(Blocks.sensitivity)+1])
%         line([0 length(Blocks.sensitivity)+1+1],[0.5 0.5],'linestyle','--','linewidth',1,'color',[0.5 0.5 0.5])
%         ylim([0 10]);
        
    end
end
end