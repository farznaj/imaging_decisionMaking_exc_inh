function [nevents, stimdur, stimrate, stimtype] = setStimRateType(alldata)
% multisens = stimtype(:,1);
% onlyvis = stimtype(:,2);
% onlyaud = stimtype(:,3);

%% set event numbers and trial types: multi-sensory, only visual, only audio
nvis = [alldata.nVisualEvents]';
naud = [alldata.nAuditoryEvents]';

% correct for couting evnum as 1 when Ieis = NaN. (you changed your protocol on 3/19/15 to correct for this, but in case you are analysing data before that date, do this.)
for i = [1, length(alldata)]
    day = simpleTokenize(alldata(i).filename, '_');
    day = day{2};
    if any(datenum(day) <= datenum('30-Mar-2015'))
        alldata_visIeis = cell(1,length(alldata));
        for tri = 1:length(alldata)
            alldata_visIeis{tri} = alldata(tri).visualIeis;
        end
        a = cellfun(@(x)sum(isnan(x)), alldata_visIeis);
        nvis(a==1) = 0;

        alldata_audIeis = cell(1,length(alldata));
        for tri = 1:length(alldata)
            alldata_audIeis{tri} = alldata(tri).auditoryIeis;
        end
        a = cellfun(@(x)sum(isnan(x)), alldata_audIeis);
        naud(a==1) = 0;

        break
    end
end

multisens = (nvis~=0 & naud~=0);
onlyvis = (nvis~=0 & naud==0);
onlyaud = (nvis==0 & naud~=0);

stimtype = [multisens, onlyvis, onlyaud];

% num_multi_vis_aud = [sum(multisens) sum(onlyvis) sum(onlyaud)];


%% set stimulus rate for each trial

% nevents shows the number of events for each trial. if it is a
% multisensory trial, it shows the average of visual and auditory events.
nevents = nanmean([nvis,naud],2);
nevents(onlyvis) = nvis(onlyvis);
nevents(onlyaud) = naud(onlyaud);
% [nvis naud nevents];

% set stimdur for each trial, ie the stimdur based on which the events were created.
stimdur = [alldata.waitDuration]';
stimdurdiff = [alldata.stimDur_diff];
stimdur(stimdurdiff~=0) = stimdurdiff(stimdurdiff~=0); % if on some trials stimdurdiff was on then use that value for stimdur.

% set stimulus rate for each trial
stimrate = nevents ./ stimdur;

if sum(~nevents)>0
    warning('There are %d trials with no vis/aud events! setting their stimrate to NaN!', sum(~nevents))
    stimrate(~nevents) = NaN; 
end

% num_LR_HR_CB_ratio = [sum(ratev < cb) sum(ratev > cb) sum(ratev == cb) sum(ratev < cb)/sum(ratev > cb)]; % LR, HR, boundary


