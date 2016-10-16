% Use this script to shift and correct cs_frtrs when frames are dropped
% from trials. I believe this occurs for the frames at the end of a trial,
% because analog channels allow you to check for the alignment of the
% start of trials, but not the end of trials.
%
% When you see abrupt changes in the fluorescence traces that happen at the
% middle of a trial, you figured it is due to frames being dropped while
% mscan either acquired or saved the imaging data. As a result cs_frtrs (ie
% the frameCount text file) wont be accurate. I believe frameCount text file
% includes the correct length of the trial because it matches the length I
% compute from behavioral data. However, it does not match the length of
% frames in the imaging data, because frames are dropped. 


%%
mouse = 'fni17';
imagingFolder = '151020'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1,2];  % 3; %1; % or tif major


%%
signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
%}
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
disp(pnev_n)
cd(fileparts(imfilename))


%%
load(pnevFileName, 'activity_man_eftMask_ch2')
load(imfilename, 'cs_frtrs')
cs_frtrs_0 = cs_frtrs;


%%
tr_miss_all = [];
nfrs_missed_all = [];
col = {'c', 'y', 'o', 'k', 'm', 'r'};


%% Identify sharp chanes in the fluorescence trace

trace = abs(diff(mean(activity_man_eftMask_ch2')));
th = mean(trace) + 3.5*std(trace);

h = figure('position', [680         741        1207         235]); 
hold on;
plot([cs_frtrs ; cs_frtrs], [min(trace) max(trace)], 'g')
plot(trace); plot([0, length(trace)],[th th], 'g')

sharp_changes = find(trace >= th);


%% Quick look at the trace to check for misalignment of the sharp changes with the trial onsets (cs_frtrs)

r2 = 0;
figure(h)
for rr = 1:floor(length(trace)/.5e4)+1
    r1 = r2;
    r2 = r1+.5e4;
    xlim([r1 r2])
%     ginput
    pause
end



%% This is a quick figure (very usefull though) that helps with identifying trials with end missings.

% when you see a range of trials at 1, look at the beginning and end of the
% 1s, those are the questionable trials.

figure;
for i = 0:40,
    v = ismember(cs_frtrs_0(2:end)-i, sharp_changes);
    plot(v),
    title(['num shifts in cs_frtrs = ', num2str(i)])
    ylabel('Does a sharp change exist at the trial-onset frame minus i')
    pause
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Correcting cs_frtrs 
% it will also trials with missing ends. It though cannot find unknown trials (ie those without a sharp change)... you will need to do this manually.



%% Find the 1st problematic trial:

%% Attempt to locate the trials with missing frames at the end

i = 0;
r = 1; % :10 % keep looping until all trials with end-frame missings are identified.

v = ismember(cs_frtrs(2:end)-i, sharp_changes);
a = abs(diff(v,10));
v1 = v + [a, nan(1,length(v)-length(a))];
v1(v1>0) = 1;
%     v1(isnan(v1)) = 0;
cr = diff(v1)~= 0  &  ~isnan(diff(v1));
tr_miss = find(cr)+1;


%%
if isempty(tr_miss)
    i = i+1;
else
    
    h2 = figure; plot(v)
    figure(h2);
    t = sprintf('Trial %d problematic', tr_miss);
    title(t)
    
    
    %% Identify number of missed frames
    
    %     last_good_tr = find(sharp_changes == cs_frtrs(tr_miss+1-1)); % last trial before things getting messed up.
    %     sh_ind = last_good_tr + find((sharp_changes(last_good_tr+1:end) - cs_frtrs(tr_miss+1))>0,1);
    
    sh_ind = find((sharp_changes - cs_frtrs(tr_miss+1))>0,1) - 1;
    
    nfrs_missed = cs_frtrs(tr_miss+1) - sharp_changes(sh_ind);
    
    
    %%
    if nfrs_missed < cs_frtrs(tr_miss+1) - cs_frtrs(tr_miss)
        
        i = 0;
        
        %% Show the problematic trial on the trace for checking and confirmation
        
        fr__cent_tr_miss = cs_frtrs(tr_miss) + (cs_frtrs(tr_miss+1) - cs_frtrs(tr_miss))/2;
        figure(h)
        xlim([fr__cent_tr_miss-.5e3 fr__cent_tr_miss+.5e3])
        t = sprintf('Trial %d problematic - %d frames missed', tr_miss, nfrs_missed);
        title(t)
        plot(cs_frtrs(tr_miss+1), max(trace), 'r*')
        
        
        %% Reset tr_miss manually if you are not happy with the automatic identification
        
        Q = input('Happy with tr_miss? ');
        
        if isempty(Q) || Q~=0 % if 0 is entered, keep moving forward
            if ~isempty(Q)
                tr_miss = Q;
            end
            tr_miss_all = [tr_miss_all, tr_miss];
            
            
            %% Correct cs_frtrs by shifting it after the problematic trial.
            
            % Make sure a sharp_change exists at the end of tr_miss when corrected for nfrs_missed
            ts = ismember(cs_frtrs(1+tr_miss) - nfrs_missed, sharp_changes);
            if ts
                cs_frtrs(tr_miss+1:end) = cs_frtrs(tr_miss+1:end) - nfrs_missed;
                
                % Show the new cs_frtrs on the trace for checking
                figure(h)
                plot([cs_frtrs ; cs_frtrs], [min(trace) max(trace)], 'color', col{r})
                
                nfrs_missed_all = [nfrs_missed_all, nfrs_missed];                
            end            
        end
    else
        i = i +1;
    end
end




%% After shifting cs_frtrs, now look for more problematic trials.

i = 0;
rr = r;

for r = 2:20 % keep looping until all trials with end-frame missings are identified.
    
    %% Attempt to locate the trials with missing frames at the end
    v = (ismember(cs_frtrs(2:end)-i, sharp_changes));
    
    v1 = boxFilter(v, 10, 2);
    %{
    vd = 10;
    a = abs(diff(v,vd));
    v1 = v + [a, nan(1,length(v)-length(a))];
    v1(v1>0) = 1;
    %     v1(isnan(v1)) = 0;
    
    v1(v1==1) = -100;
    v1(v1==0) = 1;
    v1(v1==-100) = 0;
    %}
    
    cr = diff(v1)~= 0  &  ~isnan(diff(v1));
    tr_miss = find(cr, 1)+1; %+ vd
    
    
    %
    if isempty(tr_miss)
        i = i+1;
    else
        
        h2 = figure; plot(v)
        figure(h2);
        t = sprintf('Trial %d problematic', tr_miss);
        title(t)
        
    end
    
    %% Identify number of missed frames
    
    if ~isempty(tr_miss)
        
        %         tr_miss = tr_miss + vd;
        
        %%
        %     last_good_tr = find(sharp_changes == cs_frtrs(tr_miss+1-1)); % last trial before things getting messed up.
        %     sh_ind = last_good_tr + find((sharp_changes(last_good_tr+1:end) - cs_frtrs(tr_miss+1))>0,1);
        
        sh_ind = find((sharp_changes - cs_frtrs(tr_miss+1))>0,1) - 1;
        
        nfrs_missed = cs_frtrs(tr_miss+1) - sharp_changes(sh_ind);
        
        
        %%
        if ~(nfrs_missed < cs_frtrs(tr_miss+1) - cs_frtrs(tr_miss))
            i = i +1;
        else
            
            i = 0;
            %% Show the problematic trial on the trace for checking and confirmation
            
            fr__cent_tr_miss = cs_frtrs(tr_miss) + (cs_frtrs(tr_miss+1) - cs_frtrs(tr_miss))/2;
            figure(h)
            xlim([fr__cent_tr_miss-.5e3 fr__cent_tr_miss+.5e3])
            t = sprintf('Trial %d problematic - %d frames missed', tr_miss, nfrs_missed);
            title(t)
            plot(cs_frtrs(tr_miss+1), max(trace), 'r*')
            
            
            %% Reset tr_miss manually if you are not happy with the automatic identification
            
            Q = input('Happy with tr_miss? ');
            
            if isempty(Q) || Q~=0 % if 0 is entered, keep moving forward
                if ~isempty(Q)
                    tr_miss = Q;
                end
                tr_miss_all = [tr_miss_all, tr_miss];
                
                
                %% Correct cs_frtrs by shifting it after the problematic trial.
                
                % Make sure a sharp_change exists at the end of tr_miss when corrected for nfrs_missed
                ts = ismember(cs_frtrs(1+tr_miss) - nfrs_missed, sharp_changes);
                if ts
                    cs_frtrs(tr_miss+1:end) = cs_frtrs(tr_miss+1:end) - nfrs_missed;
                    
                    % Show the new cs_frtrs on the trace for checking
                    rr = rr+1;
                    figure(h)
                    plot([cs_frtrs ; cs_frtrs], [min(trace) max(trace)], 'color', col{rr})
                    
                    nfrs_missed_all = [nfrs_missed_all, nfrs_missed];
                    
                end
            end
        end
    end
    
end



%% Take a look at the trace again; Now you should have corrected cs_frtrs

r2 = 0;
figure(h), 
plot([cs_frtrs ; cs_frtrs], [min(trace) max(trace)], 'm')

for rr = 1:floor(length(trace)/.5e4)+1
    r1 = r2;
    r2 = r1+.5e4;
    xlim([r1 r2])
    pause
end


%% Final look:

tr_miss_all

for i = 1:length(tr_miss_all)
    fr__cent_tr_miss = cs_frtrs(tr_miss_all(i)) + (cs_frtrs(tr_miss_all(i)+1) - cs_frtrs(tr_miss_all(i)))/2;
    figure(h)
    xlim([fr__cent_tr_miss-.5e3 fr__cent_tr_miss+.5e3])
    
    t = sprintf('Trial %d problematic - %d frames missed', tr_miss_all(i), nfrs_missed_all(i));
    title(t)
    plot(cs_frtrs(tr_miss_all(i)+1), max(trace), 'r*')
            
    pause
end
        

%% Final set of problematic (frame-dropped) trials

% tr_miss_all = [211   220];
   
trEndMissing = [211]; % their start is fine, so alignment on the start is fine.
trEndMissingUnknown = [215]; % no sharp change at their end end so should be fine for alignements on the start.
trStartMissingUnknown = [216:220]; % there was not a sharp change at their beginning so we don't know if they are mis-aligned... don't use them for any alignment.


%%
cs_frtrs_old = cs_frtrs_0;


%% Append the correct cs_frtrs and problematic trials to imfilename
% remember you are not correcting for Nnan variable, because you are not
% using it anymore.

save(imfilename, '-append', 'trEndMissing', 'trEndMissingUnknown', 'trStartMissingUnknown', 'cs_frtrs_old', 'cs_frtrs')







