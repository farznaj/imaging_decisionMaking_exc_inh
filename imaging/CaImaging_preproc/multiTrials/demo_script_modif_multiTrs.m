function [A, C, S, C_df, S_df, Df, b, f, srt, Ain, options, P, merging_vars] = demo_script_modif_multiTrs(Y, pnev_inputParams, cs_frtrs)
% [A, C, S, C_df, S_df, Df, b, f, srt, options, P] = demo_script_modif(Y, pnev_inputParams);
%
% INPUTSL
%     Y                 % movie (height x width x frames)
%     pnev_inputParams  % structure of input parameters with the following fields:
%         K                      % number of components to be found
%         temp_sub               % temporal subsampling for greedy initiation, set to 1 for no down sampling.
%         space_sub              % spatial subsampling for greedy initiation, set to 1 for no down sampling.
%         tau                    % std of gaussian kernel (size of neuron) 
%         p                      % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
%         merge_thr              % merging threshold
%         deconv_method          % activity deconvolution method
%         temp_iter              % number of block-coordinate descent steps 
%         fudge_factor           % bias correction for AR coefficients
%         finalRoundMCMC         % do a final round of MCMC method (if false, after merging 2 iterations of const foopsi will be done. If true, after merging 1 iter of const foopsi and 1 iter of MCMC will be done.)
%         doPlots                % make some figures and a movie.
%         parallelTempUpdate     % do parallel temporal updating.
%         save4debug             % save Eftychios's variables (eg A,C,etc) after each step for debug purposes. 
%         search_dist            % search distance when updating spatial components.
%   cs_frtrs           % cumsum of number of frames per trial recorded in movie Y. 
%
% OUTPUTS:
%     A;          % spatial components of neurons 
%     C;          % temporal components of neurons 
%     S;          % spike counts 
%     C_df;       % temporal components of neurons and background normalized by Df        
%     S_df;       % spike counts of neurons normalized by Df
%     Df;         % background for each component to normalize the filtered raw data          
%     b;          % spatial components of backgrounds
%     f;          % temporal components of backgrounds
%     srt         % index of ROIs before ordering
%     Ain         % spatial components of neurons after initialization
%     options;    % options for model fitting  
%     P;          % some estimated parameters 
%     merging_vars % some vars related to merging; allows offline assessment if desired.
%
% based on Eftychios's demo_script (repository ca_source_extraction V0.2.1)
% use processCaImagingMCPnev.m to set INPUT.
% FN (Jan 11 2016)
        

%% convert Tif to double

%{
% Read tiff files. % FN
mousename = 'fni17';
imagingFolder = '151102';
mdfFileNumber = 1; % or tif major
ch2read = 2;
  
tifList = tifListSet(mousename, imagingFolder, mdfFileNumber, ch2read);
tifList = tifList(end);
Y = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    Y = cat(3, Y, bigread2(tifList{t}));
end
%}

Y = Y - min(Y(:));
if ~isa(Y,'double');    Y = double(Y);  end         % convert to double

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels


%% Set parameters
% eg default vars:
%{
search_dist = 3;
deconv_meth = 'constrained_foopsi';
temp_iter = 2;
fudge_fact = 0.98;
merge_thr = 0.8;
tau = 4; 
K = 30; 
p = 2; 
temp_sub = 1;
space_sub = 1;

doPlots = 1;
save4debug = 0;
pnev_inputParams.save_merging_vars = 1;
finalRoundMCMC = 0;
concatTempUpdate = 1;
%}

K = pnev_inputParams.K;             % number of components to be found
tau = pnev_inputParams.tau;         % std of gaussian kernel (size of neuron); Eftychios's default=4;  
p = pnev_inputParams.p;             % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay); Eftychios's default=2;  
merge_thr = pnev_inputParams.merge_thr; % Eftychios's default=0.8;  
deconv_meth = pnev_inputParams.deconv_method;
temp_iter = pnev_inputParams.temp_iter;
fudge_fact = pnev_inputParams.fudge_factor;
temp_sub = pnev_inputParams.temp_sub;
space_sub = pnev_inputParams.space_sub;
finalRoundMCMC = pnev_inputParams.finalRoundMCMC;
doPlots = pnev_inputParams.doPlots;
% parallelTempUpdate = pnev_inputParams.parallelTempUpdate; % FN commented.
% In V0.3.3 options.temporal_parallel takes care of it and it will be by
% default true if parallel processing exists on the machine.
save4debug = pnev_inputParams.save4debug;
search_dist = pnev_inputParams.search_dist;
concatTempUpdate = pnev_inputParams.concatTempUpdate; % if true, temporal updating will be done on the movie that includes concatenated trials (datasets), otherwise the trace of each dateset (trial) will be updated separately; this gives better estimate of time constants and removes artificial spikes at the begining of each dateset (trial).  

options = CNMFSetParms(...                      
    'd1',d1,'d2',d2,...                                   % dimensions of datasets
    'search_method','ellipse','dist',search_dist,...      % search locations when updating spatial components
    'deconv_method',deconv_meth,...                       % activity deconvolution method
    'temporal_iter',temp_iter,...                         % number of block-coordinate descent steps 
    'fudge_factor',fudge_fact,...                         % bias correction for AR coefficients
    'merge_thr',merge_thr,...                             % merging threshold
    'gSig',tau,...
    'tsub', temp_sub,...                                  % temporal subsampling for greedy initiation
    'ssub', space_sub ...                                 % spatial subsampling for greedy initiation
    );


%% Data pre-processing

[P,Y] = preprocess_data(Y,p);


%% fast initialization of spatial components using greedyROI and HALS

[Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options);  % initialize

if doPlots
  % display centers of found components
  Cn =  reshape(P.sn,d1,d2); %correlation_image(Y); %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)
  figure;imagesc(Cn);
    axis equal; axis tight; hold all;
    scatter(center(:,2),center(:,1),'mo');
    title('Center of ROIs found from initialization algorithm');
    drawnow;
end

if save4debug
    nowStr = datestr(now, 'yymmdd-HHMMSS');
    save(['Eft_aftInit-', nowStr], 'Ain','bin','Cin','fin','center','P','options')
end


%% manually refine components (optional)
refine_components = false;  % flag for manual refinement
if refine_components
    [Ain,Cin,center] = manually_refine_components(Y,Ain,Cin,center,Cn,tau,options);
end


%% update spatial components

fprintf('================== Updating spatial components started ==================\n')

Yr = reshape(Y,d,T);
clear Y;
[A,b,Cin] = update_spatial_components(Yr,Cin,fin,Ain,P,options);

if save4debug
    nowStr = datestr(now, 'yymmdd-HHMMSS');
    save(['Eft_aftInitAndSpUp-', nowStr], 'A','b','Cin','fin','Ain','P','options')
end


%%
cs_frtrs = framesPerTrialMovie(mousename, imagingFolder, mdfFileNumber, 4);

frameLength = 1000/30.9; % msec.

% load alldata
% set filenames
[alldata_fileNames, ~] = setBehavFileNames(mousename, {datestr(datenum(imagingFolder, 'yymmdd'))});
% sort it
[~,fn] = fileparts(alldata_fileNames{1});
a = alldata_fileNames(cellfun(@(x)~isempty(x),cellfun(@(x)strfind(x, fn(1:end-4)), alldata_fileNames, 'uniformoutput', 0)))';
[~, isf] = sort(cellfun(@(x)x(end-25:end), a, 'uniformoutput', 0));
alldata_fileNames = alldata_fileNames(isf);
% load the one corresponding to mdffilenumber.
[all_data, ~] = loadBehavData(alldata_fileNames(mdfFileNumber)); %, defaultHelpedTrs, saveHelpedTrs); % it removes the last trial too.
fprintf('Total number of trials: %d\n', length(all_data))

iti_noscan = itiSet(all_data); % ms
iti_noscan(1) = []; % so iti_noscan(i) gives ITI following trial i.
nFrames_iti = round(iti_noscan/frameLength); % nFrames between trials (during iti) that was not imaged.
length(nFrames_iti)
nFrames_iti = nFrames_iti([100-1, 100:107]); % 100:107 are trials that correspond to movie 4. 99 was partially recorded at the beginning.
length(nFrames_iti)


%% update temporal components

fprintf('================== Updating temporal components started ==================\n')

% if parallelTempUpdate
%     try 
%         [C,f,P,S] = update_temporal_components_parallel(Yr,A,b,Cin,fin,P,options);
%     catch ME
%         disp(ME)
%         [C,f,P,S] = update_temporal_components(Yr,A,b,Cin,fin,P,options);        
%     end
% else
%     [C,f,P,S] = update_temporal_components(Yr,A,b,Cin,fin,P,options);
% end

% options.temporal_parallel = 0; % for now.
% if concatTempUpdate % do on concat movie:
    [C,f,P,S] = update_temporal_components_multiTrs_itiNaN(Yr,A,b,Cin,fin,P,options, cs_frtrs, nFrames_iti);
%     
% else  % do on each dataset (recorded continuously) separately.
%     C = NaN(size(Cin));
%     f = NaN(size(fin));
%     S = NaN(size(Cin));
%     for i = 1:length(cs_frtrs)-1
%         fprintf('==== Trial %d ====\n', i)
%         frs = cs_frtrs(i)+1 : cs_frtrs(i+1);
%         [C(:,frs),f(frs),P,S(:,frs)] = update_temporal_components(Yr(:,frs),A,b,Cin(:,frs),fin(frs),P,options);
%     end
% end
% 
% if save4debug
%     save(['Eft_preMerge-', nowStr], 'A', 'b', 'C', 'f', 'S', 'P', 'options')
% end


%% merge found components

fprintf('================== Merging components started ==================\n')

[Am,Cm,K_m,merged_ROIs,P,Sm] = merge_components(Yr,A,b,C,f,P,S,options);

display_merging = 1; % flag for displaying merging example
if doPlots && display_merging
    for i = 1: length(merged_ROIs) % randi(length(merged_ROIs));
        ln = length(merged_ROIs{i});
        figure;
        set(gcf,'Position',[300,300,(ln+2)*300,300]);
        for j = 1:ln
            subplot(1,ln+2,j); imagesc(reshape(A(:,merged_ROIs{i}(j)),d1,d2));
            title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
        end
        subplot(1,ln+2,ln+1); imagesc(reshape(Am(:,K_m-length(merged_ROIs)+i),d1,d2));
        title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight;
        subplot(1,ln+2,ln+2);
        plot(1:T,(diag(max(C(merged_ROIs{i},:),[],2))\C(merged_ROIs{i},:))');
        hold all; plot(1:T,Cm(K_m-length(merged_ROIs)+i,:)/max(Cm(K_m-length(merged_ROIs)+i,:)),'--k')
        title('Temporal Components','fontsize',16,'fontweight','bold')
        drawnow;        
    end
end

if pnev_inputParams.save_merging_vars
    merging_vars.Am = Am;
    merging_vars.Cm = Cm;
    merging_vars.K_m = K_m;
    merging_vars.merged_ROIs = merged_ROIs;
    merging_vars.A = A;
    merging_vars.C = C;
else
    merging_vars = [];
end


%% repeat: update_spatial_components

fprintf('================== Repeated updating of spatial components started ==================\n')
[A2,b2,Cm] = update_spatial_components(Yr,Cm,f,Am,P,options); 

if save4debug
    save(['Eft_aftMergeAndSpUp-', nowStr], 'A2', 'b2', 'Cm', 'f', 'Sm', 'P', 'options')
end


%% repeat: update_temporal_components

fprintf('================== Repeated updating of temporal components started ==================\n')

if ~finalRoundMCMC

%     if parallelTempUpdate
%         try
%             [C2,f2,P,S2] = update_temporal_components_parallel(Yr,A2,b2,Cm,f,P,options);
%         catch ME
%             disp(ME)
%             [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);            
%         end
%     else
%         [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);
%     end

%     if concatTempUpdate % do on concat movie:
        [C2,f2,P,S2] = update_temporal_components_multiTrs(Yr,A2,b2,Cm,f,P,options, cs_frtrs);

%     else % do on each dataset (recorded continuously) separately.
%         C2 = NaN(size(Cm));
%         f2 = NaN(size(fin));
%         S2 = NaN(size(Cm));
%         for i = 1:length(cs_frtrs)-1
%             fprintf('==== Trial %d ====\n', i)
%             frs = cs_frtrs(i)+1 : cs_frtrs(i+1);
%             [C2(:,frs),f2(frs),P,S2(:,frs)] = update_temporal_components(Yr(:,frs),A2,b2,Cm(:,frs),f(frs),P,options);
%         end
%     end
    
else % FN: do 1 round const foopsi and then another round MCMC
    
    % round of constrained foopsi
    options.temporal_iter = 1;
%     if parallelTempUpdate
%         try
%             [C2,f2,P,S2] = update_temporal_components_parallel(Yr,A2,b2,Cm,f,P,options);
%         catch ME
%             disp(ME)
%             [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);            
%         end
%     else
%         [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);
%     end    

    if concatTempUpdate % do on concat movie:
        [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);

    else % do on each dataset (recorded continuously) separately.
        C2 = NaN(size(Cm));
        f2 = NaN(size(fin));
        S2 = NaN(size(Cm));
        for i = 1:length(cs_frtrs)-1
            fprintf('==== Trial %d ====\n', i)
            frs = cs_frtrs(i)+1 : cs_frtrs(i+1);
            [C2(:,frs),f2(frs),P,S2(:,frs)] = update_temporal_components(Yr(:,frs),A2,b2,Cm(:,frs),f(frs),P,options);
        end
    end

    if save4debug
        save(['Eft_preMCMC-', nowStr], 'A2', 'b2', 'C2', 'f2', 'S2', 'P', 'options')
    end
    
    
    % round of MCMC
    options.deconv_method = 'MCMC';
    options.MCMC_B = pnev_inputParams.MCMC_B;
    options.MCMC_Nsamples = pnev_inputParams.MCMC_Nsamples;
    options.MCMC_prec = pnev_inputParams.MCMC_prec;
    fprintf('================== Final MCMC updating of temporal components started ==================\n')
%     if parallelTempUpdate
%         try
%             [C2,f2,P,S2] = update_temporal_components_parallel(Yr,A2,b2,C2,f2,P,options);
%         catch ME
%             disp(ME)
%             [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,C2,f2,P,options);            
%         end
%     else
%         [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,C2,f2,P,options);
%     end
    
    if concatTempUpdate % do on concat movie:
        [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,C2,f2,P,options);

    else % do on each dataset (recorded continuously) separately.
        for i = 1:length(cs_frtrs)-1
            fprintf('==== Trial %d ====\n', i)
            frs = cs_frtrs(i)+1 : cs_frtrs(i+1);
            [C2(:,frs),f2(frs),P,S2(:,frs)] = update_temporal_components(Yr(:,frs),A2,b2,C2(:,frs),f2(frs),P,options);
        end
    end
    
end


%% order ROIs

fprintf('================== Ordering ROIs ==================\n')
[A_or, C_or, S_or, P, srt] = order_ROIs(A2,C2,S2,P);    % order components


%% extract DF/F 

fprintf('================== Extracting DF/F ==================\n')
K_m = size(C_or, 1);
[C_df,Df,S_df] = extract_DF_F(Yr,[A_or,b2],[C_or;f2],S_or,K_m+1); % extract DF/F values (optional) % FN moved it here so C_df and S_df are also ordered.


%% do some plotting

if doPlots
    contour_threshold = 0.95;                       % amount of energy used for each component to construct contour plot
    figure;
    [Coor,json_file] = plot_contours(A_or,reshape(P.sn,d1,d2),contour_threshold,1); % contour plot of spatial footprints
%     pause; 
    %savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)
%     view_components(Yr,A_or,C_or,b2,f2,Cn,options);         % display all components
    % display components
    plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options)
end


%% make movie

if doPlots
    make_patch_video(A_or,C_or,b2,f2,Yr,Coor,options)
end


%%
A = A_or;
C = C_or;
S = S_or;
b = b2;
f = f2;


%%
%{
% a = nanmean(an,1);
a = an';

for itr = 1:length(cs_frtrs)-1
    frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
    aa{itr} = a(frs);
end

aaa = cellfun(@(x)x(1:31), aa, 'uniformoutput', 0);
a4 = cell2mat(aaa');
figure; plot(nanmean(a4, 1))

f = figure;
for i=1:size(a4,1)
subplot(2,5,i), hold on
plot(a4(i,:))
end


a = bn;
for itr = 1:length(cs_frtrs)-1
    frs = cs_frtrs(itr)+1 : cs_frtrs(itr+1);
    aa{itr} = a(frs);
end

aaa = cellfun(@(x)x(1:31), aa, 'uniformoutput', 0);
a4 = cell2mat(aaa');
figure; plot(nanmean(a4, 1))


f = figure;
for i=1:size(a4,1)
subplot(2,5,i), hold on
plot(a4(i,:))
end


%}

