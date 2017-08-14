% Use setMouseDay to set vars needed here.

%% Compute HR,LR selectiviy on trial-averaged single neuron responses.

init_spec = cell(1,length(days));
init_time = cell(1,length(days));
stim_spec = cell(1,length(days));
stim_time = cell(1,length(days));
go_spec = cell(1,length(days));
go_time = cell(1,length(days));
choice_spec = cell(1,length(days));
choice_time = cell(1,length(days));
rew_spec = cell(1,length(days));
rew_time = cell(1,length(days));
incorr_spec = cell(1,length(days));
incorr_time = cell(1,length(days));
nHrLr = cell(1,length(days));

% tic
for iday = 1:length(days)
    
    disp('__________________________________________________________________')
    dn = simpleTokenize(days{iday}, '_');
    
    imagingFolder = dn{1};
    mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
    fprintf('Analyzing day %s, sessions %s\n', imagingFolder, dn{2})
    
    
    %%
%     try
        [ispec, itimed, sspec, stimed, gspec, gtimed, cspec, ctimed, rspec, rtimed, pspec, ptimed, ni1,ni0,ns1,ns0,ng1,ng0,nc1,nc0,nr1,nr0,np1,np0]...
            = selHrLr(mouse, imagingFolder, mdfFileNumber);
        
        init_spec{iday} = ispec;
        init_time{iday} = itimed;
        
        stim_spec{iday} = sspec;
        stim_time{iday} = stimed;
        
        go_spec{iday} = gspec;
        go_time{iday} = gtimed;
        
        choice_spec{iday} = cspec;
        choice_time{iday} = ctimed;
        
        rew_spec{iday} = rspec;
        rew_time{iday} = rtimed;
        
        incorr_spec{iday} = pspec;
        incorr_time{iday} = ptimed;
        
        nHrLr{iday} = [ni1,ni0; ns1,ns0; ng1,ng0; nc1,nc0; nr1,nr0; np1,np0];
        
%     catch ME
%         disp(ME)
% %         disp(ME.stack)
%     end
end



%% Align selectivity traces of all days

[init_spec_aligned, stim_spec_aligned, go_spec_aligned, choice_spec_aligned, rew_spec_aligned, incorr_spec_aligned, ...
    time_alignedi, time_aligneds, time_alignedg, time_alignedc, time_alignedr, time_alignedp] ...
= selHrLrAlign(init_time, stim_time, go_time, choice_time, rew_time, incorr_time, ...
    init_spec, stim_spec, go_spec, choice_spec, rew_spec, incorr_spec);


%% Prep plotting

tr = {init_spec_aligned, stim_spec_aligned, go_spec_aligned, choice_spec_aligned, rew_spec_aligned, incorr_spec_aligned};
t = {time_alignedi, time_aligneds, time_alignedg, time_alignedc, time_alignedr, time_alignedp};
lab = {'init', 'stim', 'go', 'choice', 'reward', 'incorrResp'};


%% We need at least 10 trials for each HR and LR categories; otherwise set aligned traces for days that dont meet this condition to nan.

th = 10;
alFew = cell2mat(cellfun(@(x)sum(x<th,2), nHrLr, 'uniformoutput', 0)); % each row is for one of the alignments. if 
%{
for i = 1:length(t) % different alignments
    tr{i}(:, alFew(i,:)~=0) = nan;
end
%}



%%
%%%% PLOTS

%%
figure('name', 'ave+/-sd across bootstrap samples');
for i = 1:length(t)
    if sum(alFew(i,:),2) < length(days)*2 % dont try to plot incorrAl when analyzing only correct trials.
        el = find(sign(t{i})>0,1); % eventI on the aligned traces

        pre_c = squeeze(mean(tr{i}(el-3:el-1 , :, :),1)); %samp x days % remember el includes frame 0, so if there are some very early responses to event, then quantity computed here is not quite the baseline.
        post_c = squeeze(mean(tr{i}(el+1:min(el+3,size(tr{i},1)) , :, :),1)); % i'm just concerned about the decrease in S at the end of trials... so maybe I shoudln't go upto the end.

        subplot(3,2,i), hold on; 

        m = mean(pre_c,1); %m(isnan(m)) = []; 
        s = std(pre_c,[],1); %s(isnan(s)) = [];
        [h1,h2] = boundedline(1:length(m), m, s, 'alpha', 'nan', 'gap'); 
        set(h1, 'color', 'b')
        set(h2, 'facecolor', 'b')
    %     xm = 1:size(pre_c,2);
    %     set(gca, 'xtick', xm)
    %     set(gca, 'xticklabel', xm(~isnan(mean(pre_c,1))))
    %     xtickangle(45)

        m = mean(post_c,1); %m(isnan(m)) = []; 
        s = std(post_c,[],1); %s(isnan(s)) = [];
        [h1, h2] = boundedline(1:length(m), m, s, 'alpha', 'nan', 'gap'); 
        set(h1, 'color', 'r')
        set(h2, 'facecolor', 'r')

        plot([1,length(days)], [0,0], ':')
    end
    title(lab{i})
end
subplot(3,2,1)
xlabel('days')
ylabel('(hr-lr) / (hr+lr)')


%% each day in a subplot

r = 7; c = 7;
for i = 1:length(t) % different alignments
    figure; set(gcf,'name',lab{i})
    for j=1:size(tr{i},2) % days        
        subplot(r,c,j)
        plot(t{i},tr{i}(:,j))
        title(days{j})
    end
end

%% all days superimposed

co = hot(length(days)*2+2);
co = co(1:2:end,:);
set(groot,'defaultAxesColorOrder',co)

figure; 
for i = 1:length(t) % different alignments    
    subplot(3,2,i)
    hold on
    plot(t{i},tr{i})    
    title(lab{i})
end
% legend(days)

%% abs of selectivity

figure; 
for i = 1:length(t) % different alignments    
    subplot(3,2,i)
    hold on
    plot(t{i}, abs(tr{i}))    
    title(lab{i})
end

%%
set(groot,'defaultAxesColorOrder','remove')

