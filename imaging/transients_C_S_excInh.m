
savFigs = 0;
bins = [.4:.05:1]; % bins for binning spike values
bins(end) = 1.1;
fbef = 20; % transients of C and S will have fbef frames before each spike and faft frames after each spike
faft = 80;
fbef_st = 10; % number of frames before and after each spike that we want to be spike-free (ie S<st_bl)
st_bl = .05; %.2; % magnitude of S of a spike in order to call it spike-free 
% spVal = .4;  % lets pick the spikes in bin i perhaps we dont want to take spikes < .05 into account ... so perhaps dont worry about spikes in those bins


mice = {'fni16','fni17','fni18','fni19'};
dirn0tr = '/home/farznaj/Dropbox/ChurchlandLab/Projects/inhExcDecisionMaking/Transients_inh_exc';
ColOrd = get(gca,'ColorOrder'); close
nowStr = datestr(now, 'yymmdd-HHMMSS');


%%
Cinh_allMice = cell(1, length(mice));
Cexc_allMice = cell(1, length(mice));
Sinh_allMice = cell(1, length(mice));
Sexc_allMice = cell(1, length(mice));

% im = 1;
for im = 1:length(mice)
    
    mouse = mice{im};
    
    %%% Set days for each mouse
    
    if strcmp(mouse, 'fni16')
        days = {'150817_1', '150818_1', '150819_1', '150820_1', '150821_1-2', '150825_1-2-3', '150826_1', '150827_1', '150828_1-2', '150831_1-2', '150901_1', '150903_1', '150904_1', '150915_1', '150916_1-2', '150917_1', '150918_1-2-3-4', '150921_1', '150922_1', '150923_1', '150924_1', '150925_1-2-3', '150928_1-2', '150929_1-2', '150930_1-2', '151001_1', '151002_1-2', '151005_1-2-3-4', '151006_1-2', '151007_1-2', '151008_1', '151009_1', '151012_1-2', '151013_1', '151014_1-2', '151016_1', '151019_1', '151020_1', '151021_1', '151022_1', '151023_1', '151026_1-2', '151027_1', '151028_1-2', '151029_1-2'}; %'150914_1-2' : don't analyze!
    elseif strcmp(mouse, 'fni17')
        days = {'150814_1', '150817_1', '150824_1', '150826_1', '150827_1', '150828_1', '150831_1', '150901_1', '150902_1-2', '150903_1', '150908_1', '150909_1', '150910_1', '150914_1', '150915_1-2', '150916_1', '150917_1-2', '150918_1', '150921_1-2-3', '150922_1-2', '150923_1-2-3', '150924_1-2', '150925_1-2', '150928_1-2', '150930_1-2-3-4', '151001_1', '151002_1-2', '151005_1-2', '151006_1', '151007_1', '151008_1', '151010_1', '151012_1-2-3', '151013_1-2', '151014_1', '151015_1', '151016_1', '151019_1-2', '151020_1-2', '151021_1', '151022_1-2', '151023_1', '151026_1', '151027_2', '151028_1-2-3', '151029_2-3', '151101_1', '151102_1-2'};
    elseif strcmp(mouse, 'fni18')
        days = {'151209_1', '151210_1', '151211_1', '151214_1-2', '151215_1-2', '151216_1', '151217_1-2'}; % alldays
    elseif strcmp(mouse, 'fni19')
        days = {'150903_1', '150904_1', '150914_1', '150915_1', '150916_1', '150917_1', '150918_1', '150921_1', '150922_1', '150923_1', '150924_1-2', '150925_1-2', '150928_4', '150929_3', '150930_1', '151001_1', '151002_1', '151005_1-2', '151006_1', '151007_1', '151008_1-2', '151009_1-3', '151012_1-2-3', '151013_1', '151015_2', '151016_1', '151019_1', '151020_1', '151022_1-2', '151023_1', '151026_1-2-3', '151027_1', '151028_1-2', '151029_1-2-3', '151101_1'};
    end
    
    
    %%
    % iday = 1;
    for iday = 1:length(days)
        
        disp('__________________________________________________________________')
        dn = simpleTokenize(days{iday}, '_');
        
        imagingFolder = dn{1};
        mdfFileNumber = str2double(simpleTokenize(dn{2}, '-'));
        
        fprintf('Analyzing %s, day %d/%d (%s, sessions %s)\n', mouse, iday, length(days), imagingFolder, dn{2})
        
        signalCh = 2; % because you get A from channel 2, I think this should be always 2.
        pnev2load = [];
        [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
        [pd, pnev_n] = fileparts(pnevFileName);
        moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n));
        postName = fullfile(pd, sprintf('post_%s.mat', pnev_n));
        
        
        %%
        clear C S cs_frtrs badROIs01 %C0 S0
        load(pnevFileName, 'C', 'S')
        load(imfilename, 'cs_frtrs')
        load(moreName, 'inhibitRois_pix', 'badROIs01')
        
        C = C(~badROIs01,:);
        S = S(~badROIs01,:);
%         C0 = C;
%         S0 = S;
                
        
        %% Set some frames to nan and normalize S and C(?) of each neuron by its max.

        % Set ~20 frames at the begining of each trial to nan, bc C will be
        % high in those frames due to spikes that happened before imaging
        % onset.
        for i = 1:length(cs_frtrs)-1
            S(:, cs_frtrs(i)+1 : min(size(S,2), cs_frtrs(i)+10)) = nan; %20, 100
            C(:, cs_frtrs(i)+1 : min(size(S,2), cs_frtrs(i)+10)) = nan;
        end
        
        % do the normalization
        S = bsxfun(@rdivide, S, max(S,[],2));
        C = bsxfun(@rdivide, C, max(C,[],2));
                
        %{
        % first set frames around trial onsets to nan. We will not analyze spikes that happen at these frames, so we dont want them to affect normalization of C and S.        
        for i = 2:length(cs_frtrs)-1
            S(:, max(1,cs_frtrs(i)-faft) : min(size(S,2), cs_frtrs(i)+fbef)) = nan;
            C(:, max(1,cs_frtrs(i)-faft) : min(size(S,2), cs_frtrs(i)+fbef)) = nan;
        end
        % first tr
        S(:, 1:fbef) = nan;
        C(:, 1:fbef) = nan;
        % last tr
        S(:, end-faft+1:end) = nan;
        C(:, end-faft+1:end) = nan;        
        
        % do the normalization
        S = bsxfun(@rdivide, S, max(S,[],2));
        C = bsxfun(@rdivide, C, max(C,[],2));
        %}

        
        %% Find spikes of certain amplitude and set C and S around them (set the "transients")
        
        % we want to make sure we get to see at least 10 frames after a
        % spike... so set 10 frames before trial onsets to nan before we
        % find spikes of certain amplitude (using histcounts)
        for i = 2:length(cs_frtrs)-1           
            S(:, max(1,cs_frtrs(i)-10) : cs_frtrs(i)) = nan;
            C(:, max(1,cs_frtrs(i)-10) : cs_frtrs(i)) = nan;
        end
        
        
        % bin spikes    
        [n, ~, b] = histcounts(S, bins);
        
        % find spikes of certain amplitude
%         bi = find(bins==spVal); % bins(bi)
        
        %%%%%% loop over bins of S value        
        for bi = 1:length(bins)-1        

            %%%%% loop over neurons
            Cnow = cell(1, size(C,1));
            Snow = cell(1, size(C,1));
            for ni = 1:size(C,1)           
                % identify spikes that meet the two criteria below
                % 1. spike magnitude in bin bi of bins
                st = find(b(ni,:)==bi); % all spike times whose spike magnitudes are in bin bi of bins  % st = find(ismember(b(ni,:), [bi:bi+5])); 
                
                % 2. spikes that are not preceeded or followed by other spikes in 10 frames 
                cnt = 0;
                for sti = 1:length(st)
                    if all(S(ni, max(1, st(sti)-fbef_st) : st(sti)-1) < st_bl) && all(S(ni, st(sti)+1 : min(size(S,2), st(sti)+fbef_st)) < st_bl) % only pick spikes that are not preceeded or followed by other spikes in 10 frames 
                        cnt = cnt+1;
                    end
                end
%                 disp([length(st), cnt])

                % now set the transients for the spikes found above 
                Cnow{ni} = nan(cnt, fbef+faft+1);
                Snow{ni} = nan(cnt, fbef+faft+1);            
                for cnti = 1:cnt
                    r = max(1, st(cnti)-fbef) : min(size(S,2), st(cnti)+faft);
                    Cnow{ni}(cnti,1:length(r)) = C(ni, r);
                    Snow{ni}(cnti,1:length(r)) = S(ni, r);
                end          
            end
%             disp(cellfun(@(x)size(x,1), Cnow))
            
            
            %% Set C and S transients (with S value in bin bi) for exc and inh neurons

            % C
            Cinh_allMice{im}{iday}{bi} = Cnow(inhibitRois_pix==1);
            Cexc_allMice{im}{iday}{bi} = Cnow(inhibitRois_pix==0);        

            % S
            Sinh_allMice{im}{iday}{bi} = Snow(inhibitRois_pix==1);
            Sexc_allMice{im}{iday}{bi} = Snow(inhibitRois_pix==0);        
            
        end
    
        %{
        Cinh = cell2mat(Cinh0');
        Cexc = cell2mat(Cexc0');
        
        figure('position', [32   136   512   824]);
        subplot(211); hold on;
        plot(nanmean(Cexc,1), 'k')
        plot(nanmean(Cinh,1), 'r')
        
        subplot(212); hold on;
        plot(nanmean(Sexc,1), 'k')
        plot(nanmean(Sinh,1), 'r')
        %}
        
    end
end

no

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Pool "transients" across all neurons for each day

Cinh_eachDay = cell(1,length(mice));
Cexc_eachDay = cell(1,length(mice));
Sinh_eachDay = cell(1,length(mice));
Sexc_eachDay = cell(1,length(mice));
% Cinh_allDaysPooled = cell(1,length(mice));
% Cexc_allDaysPooled = cell(1,length(mice));
% Sinh_allDaysPooled = cell(1,length(mice));
% Sexc_allDaysPooled = cell(1,length(mice));

for im = 1:length(mice)
    for ibin = 1:length(bins)-1
        for iday = 1:length(Cinh_allMice{im})
            % C
            Cinh0 = Cinh_allMice{im}{iday}{ibin}; % each cell includes transients of one neuron
            Cexc0 = Cexc_allMice{im}{iday}{ibin};        
            % for each day pool transients of all neurons 
            Cinh_eachDay{im}{iday}{ibin} = cell2mat(Cinh0'); 
            Cexc_eachDay{im}{iday}{ibin} = cell2mat(Cexc0');                

            % S
            Sinh0 = Sinh_allMice{im}{iday}{ibin}; % each cell belongs to one neuron and includes all calcium transients for that neuron.
            Sexc0 = Sexc_allMice{im}{iday}{ibin};
            % for each day pool transients of all neurons 
            Sinh_eachDay{im}{iday}{ibin} = cell2mat(Sinh0'); % pool all calcium transients of all neurons
            Sexc_eachDay{im}{iday}{ibin} = cell2mat(Sexc0');
        end
    end
    % pool transients of all neurons across all days
    %{
    Cinh_allDaysPooled{im} = cell2mat(Cinh_eachDay{im}');
    Cexc_allDaysPooled{im} = cell2mat(Cexc_eachDay{im}');

    Sinh_allDaysPooled{im} = cell2mat(Sinh_eachDay{im}');
    Sexc_allDaysPooled{im} = cell2mat(Sexc_eachDay{im}');    
    %}
end


%% For each bin, pool transients of all days

Cinh_allTransientsAllDaysPooled{im}{ibin} = cell(1,length(mice));
Cexc_allTransientsAllDaysPooled{im}{ibin} = cell(1,length(mice));
Sinh_allTransientsAllDaysPooled{im}{ibin} = cell(1,length(mice));
Sexc_allTransientsAllDaysPooled{im}{ibin} = cell(1,length(mice));

for im = 1:length(mice)
    for ibin = 1:length(bins)-1                
        Cinh_allTransientsAllDaysPooled{im}{ibin} = []; 
        Cexc_allTransientsAllDaysPooled{im}{ibin} = []; 
        Sinh_allTransientsAllDaysPooled{im}{ibin} = []; 
        Sexc_allTransientsAllDaysPooled{im}{ibin} = [];          
        
        for iday = 1:length(Cinh_eachDay{im})            
            % C
            Cinh_allTransientsAllDaysPooled{im}{ibin} = [Cinh_allTransientsAllDaysPooled{im}{ibin}; Cinh_eachDay{im}{iday}{ibin}]; % number_pooled_transients x frs
            Cexc_allTransientsAllDaysPooled{im}{ibin} = [Cexc_allTransientsAllDaysPooled{im}{ibin}; Cexc_eachDay{im}{iday}{ibin}];
            
            % S
            Sinh_allTransientsAllDaysPooled{im}{ibin} = [Sinh_allTransientsAllDaysPooled{im}{ibin}; Sinh_eachDay{im}{iday}{ibin}];
            Sexc_allTransientsAllDaysPooled{im}{ibin} = [Sexc_allTransientsAllDaysPooled{im}{ibin}; Sexc_eachDay{im}{iday}{ibin}];            
        end
    end
end


%%%% For each mouse, average across pooled transients of all days

Cinh_avePooledTransients = nan(fbef+faft+1, length(bins)-1, length(mice));
Cexc_avePooledTransients = nan(fbef+faft+1, length(bins)-1, length(mice));
Sinh_avePooledTransients = nan(fbef+faft+1, length(bins)-1, length(mice));
Sexc_avePooledTransients = nan(fbef+faft+1, length(bins)-1, length(mice));

for im = 1:length(mice)
    % C
    Cinh_avePooledTransients(:,:,im) = cell2mat(cellfun(@(x)nanmean(x,1), Cinh_allTransientsAllDaysPooled{im}, 'UniformOutput', 0)')';
    Cexc_avePooledTransients(:,:,im) = cell2mat(cellfun(@(x)nanmean(x,1), Cexc_allTransientsAllDaysPooled{im}, 'UniformOutput', 0)')';
    
    % S
    Sinh_avePooledTransients(:,:,im) = cell2mat(cellfun(@(x)nanmean(x,1), Sinh_allTransientsAllDaysPooled{im}, 'UniformOutput', 0)')';
    Sexc_avePooledTransients(:,:,im) = cell2mat(cellfun(@(x)nanmean(x,1), Sexc_allTransientsAllDaysPooled{im}, 'UniformOutput', 0)')';
end



%%%%%% All mice: plot average across days, inh vs exc

x = (1:fbef+faft+1)-(fbef+1);

for ibin = 1:length(bins)-1
    spVal = bins(ibin);
    figure('name', ['All mice', ', S: ', num2str(bins(ibin))], 'position',[15         478        1260         481]); 
    
    for im = 1:length(mice)
        % C
        subplot(2,length(mice),im); hold on
        plot(x, Cinh_avePooledTransients(:,ibin,im), 'r')
        plot(x, Cexc_avePooledTransients(:,ibin,im), 'k')
        legend('inh','exc')    
        title(mice{im})
        if im==1
            ylabel('C')
            xlabel('Frames relative to spike')    
        end

        % S
        subplot(2,length(mice),im+length(mice)); hold on
        plot(x, Sinh_avePooledTransients(:,ibin,im), 'r')
        plot(x, Sexc_avePooledTransients(:,ibin,im), 'k')
        legend('inh','exc')    
        title(mice{im})    
        if im==1
            ylabel('S')
            xlabel('Frames relative to spike')    
        end
    end


    if savFigs
        namv = sprintf('transients_C_S_aveDays_allMice_Sbin0%.0f_%s', spVal*10, nowStr);

        d = fullfile(dirn0tr, 'sumAllMice');
        if ~exist(d,'dir')
            mkdir(d)
        end    
        fn = fullfile(d, namv);

        savefig(gcf, fn)
    end
end




%% Average transients for each day and compute averages across days for each mouse (do this for each bin of S)

Cinh_eachDayAved = cell(1,length(mice));
Cexc_eachDayAved = cell(1,length(mice));
Sinh_eachDayAved = cell(1,length(mice));
Sexc_eachDayAved = cell(1,length(mice));
for im = 1:length(mice)
    % C
    % for each day, average transients in each bin
    for iday = 1:length(Cinh_eachDay{im})
        Cinh_eachDayAved{im}(:,:,iday) = cell2mat(cellfun(@(x)nanmean(x,1), Cinh_eachDay{im}{iday}, 'uniformoutput', 0)')'; % frs x bins x days %%%% frs x days
        Cexc_eachDayAved{im}(:,:,iday) = cell2mat(cellfun(@(x)nanmean(x,1), Cexc_eachDay{im}{iday}, 'uniformoutput', 0)')';    
    
        % S
        % for each day, average transients in each bin
        Sinh_eachDayAved{im}(:,:,iday) = cell2mat(cellfun(@(x)nanmean(x,1), Sinh_eachDay{im}{iday}, 'uniformoutput', 0)')'; % frs x days
        Sexc_eachDayAved{im}(:,:,iday) = cell2mat(cellfun(@(x)nanmean(x,1), Sexc_eachDay{im}{iday}, 'uniformoutput', 0)')';    
    end
end


%% for each bin, average across days

Cinh_aveAllDays = nan(fbef+faft+1, length(bins)-1, length(mice));
Cexc_aveAllDays = nan(fbef+faft+1, length(bins)-1, length(mice));
Sinh_aveAllDays = nan(fbef+faft+1, length(bins)-1, length(mice));
Sexc_aveAllDays = nan(fbef+faft+1, length(bins)-1, length(mice));

for im = 1:length(mice)
    % C
    Cinh_aveAllDays(:,:,im) = nanmean(Cinh_eachDayAved{im},3); % frames x bins x mice
    Cexc_aveAllDays(:,:,im) = nanmean(Cexc_eachDayAved{im},3);

    % S
    Sinh_aveAllDays(:,:,im) = nanmean(Sinh_eachDayAved{im},3);
    Sexc_aveAllDays(:,:,im) = nanmean(Sexc_eachDayAved{im},3);    
end


%% Average across mice (for each bin of S)

% C
Cinh_aveMice = nanmean(Cinh_aveAllDays,3); % frames x bins
Cexc_aveMice = nanmean(Cexc_aveAllDays,3);
% S
Sinh_aveMice = nanmean(Sinh_aveAllDays,3);
Sexc_aveMice = nanmean(Sexc_aveAllDays,3);




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% PLOTS %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot: each mouse: individual days and average across days

for ibin = 1:length(bins)-1
    spVal = bins(ibin);
    
    for im = 1:length(mice)
        m = size(Cinh_eachDayAved{im},3);    
        col = magma(m);
    %     col = inferno(m);
    %     col = plasma(m);
    %     col = viridis(m);
    %     col = copper(m,1));   

        % C
        figure('name', [mice{im}, ', S: ', num2str(bins(ibin))], 'position', [1         424        1018         541]); 
        set(gcf, 'defaultaxescolororder', col)

        subplot(231)
        plot(x, squeeze(Cinh_eachDayAved{im}(:,ibin,:)))
        title('inh (all days)')    
        ylabel('C')
        xlabel('Frames relative to spike')    
        subplot(232)
        plot(x, squeeze(Cexc_eachDayAved{im}(:,ibin,:)))
        title('exc (all days)')    
        % average across days
        subplot(233); hold on
        plot(x, Cinh_aveAllDays(:,ibin,im), 'r')
        plot(x, Cexc_aveAllDays(:,ibin,im), 'k')
        legend('inh','exc')  
        title('mean days')


        % S
    %     figure('name', mice{im}, 'position', [45   327   678   538]); 
    %     set(gcf, 'defaultaxescolororder', col)    
        subplot(234)
        plot(x, squeeze(Sinh_eachDayAved{im}(:,ibin,:)))
        title('inh')    
        ylabel('S')
        xlabel('Frames relative to spike')    
        subplot(235)
        plot(x, squeeze(Sexc_eachDayAved{im}(:,ibin,:)))
        title('exc')    
        % average across days
        subplot(236); hold on
        plot(x, Sinh_aveAllDays(:,ibin,im), 'r')
        plot(x, Sexc_aveAllDays(:,ibin,im), 'k')
        legend('inh','exc')        


        if savFigs
            namv = sprintf('transients_C_S_allDays_Sbin0%.0f_%s_%s', spVal*10, mice{im}, nowStr);

            d = fullfile(dirn0tr, mice{im});
            if ~exist(d,'dir')
                mkdir(d)
            end
            fn = fullfile(d, namv);

            savefig(gcf, fn)
        end
    end
end


%% All mice: plot average across days, inh vs exc

for ibin = 1:length(bins)-1
    spVal = bins(ibin);
    figure('name', ['All mice', ', S: ', num2str(bins(ibin))], 'position',[15         478        1260         481]); 
    
    for im = 1:length(mice)
        % C
        subplot(2,length(mice),im); hold on
        plot(x, Cinh_aveAllDays(:,ibin,im), 'r')
        plot(x, Cexc_aveAllDays(:,ibin,im), 'k')
        legend('inh','exc')    
        title(mice{im})
        if im==1
            ylabel('C')
            xlabel('Frames relative to spike')    
        end

        % S
        subplot(2,length(mice),im+length(mice)); hold on
        plot(x, Sinh_aveAllDays(:,ibin,im), 'r')
        plot(x, Sexc_aveAllDays(:,ibin,im), 'k')
        legend('inh','exc')    
        title(mice{im})    
        if im==1
            ylabel('S')
            xlabel('Frames relative to spike')    
        end
    end


    if savFigs
        namv = sprintf('transients_C_S_aveDays_allMice_Sbin0%.0f_%s', spVal*10, nowStr);

        d = fullfile(dirn0tr, 'sumAllMice');
        if ~exist(d,'dir')
            mkdir(d)
        end    
        fn = fullfile(d, namv);

        savefig(gcf, fn)
    end
end



%% Plot average across mice: inh vs exc
DO THIS FOR EACH MOUSE ... ALSO ADD THIS FOR POOOLED TRANSIENTS OF ALL DAYS 
    
figure('position', [121   365   332   611])

subplot(211), hold on
plot(x, Cinh_aveMice, 'r')
plot(x, Cexc_aveMice, 'k')
legend('inh','exc')    
ylabel('C')
xlabel('Frames relative to spike')

subplot(212), hold on
plot(x, Sinh_aveMice, 'r')
plot(x, Sexc_aveMice, 'k')
ylabel('S')
xlabel('Frames relative to spike')


if savFigs
    namv = sprintf('transients_C_S_aveMice_Sbin0%.0f_%s', spVal*10, nowStr);
        
    d = fullfile(dirn0tr, 'sumAllMice');
    fn = fullfile(d, namv);
    
    savefig(gcf, fn)
end


%%
%{
nBins = 100;

% set the bins
r1 = 0;
r2 = 1;
bins = r1 : (r2-r1)/nBins : r2;
bins(end) = r2 + .001;

% set x for plotting hists as the center of bins
x = mode(bins(2)-bins(1))/2 + bins; x = x(1:end-1);


%%
frBinInds_E_allMice = cell(1,length(mice));
frBinInds_I_allMice = cell(1,length(mice));

for im = 1:length(mice)
    for iday = 1:numDaysAll(im)
        if mnTrNum_allMice{im}(iday) >= thMinTrs
            
            % Firing rate of E_ipsi, etc neurons at timebin -1
            frE = squeeze(fr_exc_al_allMice{im}{iday}(1:fr2an_FR,:,:)); % exc units x trials
            frI = squeeze(fr_inh_al_allMice{im}{iday}(fr2an_FR,:,:)); % inh units x trials
            
            % Bin spike values for exc and inh
            [n_E0, ~, b_E] = histcounts(frE, bins);
            [n_I0, ~, b_I] = histcounts(frI, bins);
%{
            n_E = n_E0/sum(n_E0);
            n_I = n_I0/sum(n_I0);
            
            figure(); hold on
            plot(x, n_E, 'k')
            plot(x, n_I, 'r')
%}
            frBinInds_E_allMice{im}{iday} = b_E;
            frBinInds_I_allMice{im}{iday} = b_I;
        end
    end
end
%}

