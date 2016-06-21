function [A, C, S, C_df, S_df, Df, b, f, srt, Ain, options, P, merging_vars] = demo_script_finalMCMC(Y, pnev_inputParams)
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
%
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
%
% based on Eftychios's demo_script (repository ca_source_extraction V0.2.1)
% FN (Jan 11 2016)
        

%% convert Tif to double

%{
% Read tiff files. % FN
mousename = 'fni17';
imagingFolder = '151102';
mdfFileNumber = 1; % or tif major
ch2read = 2;
  
tifList = tifListSet(mousename, imagingFolder, mdfFileNumber, ch2read);
Y = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    Y = cat(3, Y, bigread2(tifList{t}));
end
%}

if ~isa(Y,'double');    Y = double(Y);  end         % convert to double

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels


%% Set parameters
%{
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
save4debug = pnev_inputParams.save4debug;
%}
doPlots = pnev_inputParams.doPlots;
parallelTempUpdate = pnev_inputParams.parallelTempUpdate;


%{
options = CNMFSetParms(...                      
    'd1',d1,'d2',d2,...                         % dimensions of datasets
    'search_method','ellipse','dist',3,...      % search locations when updating spatial components
    'deconv_method',deconv_meth,...             % activity deconvolution method
    'temporal_iter',temp_iter,...               % number of block-coordinate descent steps 
    'fudge_factor',fudge_fact,...               % bias correction for AR coefficients
    'merge_thr',merge_thr,...                   % merging threshold
    'tsub', temp_sub,...                        % temporal subsampling for greedy initiation
    'ssub', space_sub ...                       % spatial subsampling for greedy initiation
    );
%}

%%
Yr = reshape(Y,d,T);
clear Y;


%%
mousename = 'fni17';
imagingFolder = '151102'; % '151021';
mdfFileNumber = 1; % or tif major

signalCh = 2; % CC, mask, etc are driven from signalCh_meth1 (usually you use this as Ref, but you can change in pnev_manual_comp_match)

[~, pnevFileName] = setImagingAnalysisNames(mousename, imagingFolder, ...
    mdfFileNumber, signalCh);
disp(pnevFileName)

load(pnevFileName, 'A', 'b', 'C', 'f', 'P', 'options')
A2 = A;
b2 = b;
C2 = C;
f2 = f;
clear A b C f

Ain = [];
merging_vars = [];


%%
%{
%% Data pre-processing

[P,Y] = preprocess_data(Y,p);


%% fast initialization of spatial components using greedyROI and HALS

[Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options);  % initialize

if doPlots
% display centers of found components
Cn =  correlation_image(Y); %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)
figure;imagesc(Cn);
    axis equal; axis tight; hold all;
    scatter(center(:,2),center(:,1),'mo');
    title('Center of ROIs found from initialization algorithm');
    drawnow;
end


%% update spatial components

fprintf('Updating spatial components started.\n')

Yr = reshape(Y,d,T);
clear Y;
[A,b] = update_spatial_components(Yr,Cin,fin,Ain,P,options);

if save4debug
    nowStr = datestr(now, 'yymmdd-HHMMSS');
    save(['Eft_aftInitAndSpUp-', nowStr], 'A','b','Cin','fin','P','options')
end


%% update temporal components

fprintf('Updating temporal components started.\n')

if parallelTempUpdate
    try 
        [C,f,P,S] = update_temporal_components_parallel(Yr,A,b,Cin,fin,P,options);
    catch ME
        disp(ME)
        [C,f,P,S] = update_temporal_components(Yr,A,b,Cin,fin,P,options);        
    end
else
    [C,f,P,S] = update_temporal_components(Yr,A,b,Cin,fin,P,options);
end

if save4debug
    save(['Eft_preMerge-', nowStr], 'A', 'b', 'C', 'f', 'S', 'P', 'options')
end


%% merge found components

fprintf('Merging components started.\n')

[Am,Cm,K_m,merged_ROIs,P,Sm] = merge_components(Yr,A,b,C,f,P,S,options);

display_merging = 1; % flag for displaying merging example
if doPlots && display_merging
    i = 1; randi(length(merged_ROIs));
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


%% repeat: update_spatial_components

fprintf('Repeated updating of spatial components started.\n')
[A2,b2] = update_spatial_components(Yr,Cm,f,Am,P,options); 

if save4debug
    save(['Eft_aftMergeAndSpUp-', nowStr], 'A2', 'b2', 'Cm', 'f', 'Sm', 'P', 'options')
end
%}

%% repeat: update_temporal_components
%{
fprintf('Repeated updating of temporal components started.\n')

if ~finalRoundMCMC

    if parallelTempUpdate
        try
            [C2,f2,P,S2] = update_temporal_components_parallel(Yr,A2,b2,Cm,f,P,options);
        catch ME
            disp(ME)
            [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);            
        end
    else
        [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);
    end

else % FN: do 1 round const foopsi and then another round MCMC
    %}
    
    options.temporal_iter = 1;
    %{
    if parallelTempUpdate
        try
            [C2,f2,P,S2] = update_temporal_components_parallel(Yr,A2,b2,Cm,f,P,options);
        catch ME
            disp(ME)
            [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);            
        end
    else
        [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,Cm,f,P,options);
    end
    
    if save4debug
        save(['Eft_preMCMC-', nowStr], 'A2', 'b2', 'C2', 'f2', 'S2', 'P', 'options')
    end
    %}
    
    options.deconv_method = 'MCMC';
    options.MCMC_B = pnev_inputParams.MCMC_B;
    options.MCMC_Nsamples = pnev_inputParams.MCMC_Nsamples;
    options.MCMC_prec = pnev_inputParams.MCMC_prec;
    fprintf('Final MCMC updating of temporal components started.\n')
    if parallelTempUpdate
        try
            [C2,f2,P,S2] = update_temporal_components_parallel(Yr,A2,b2,C2,f2,P,options);
        catch ME
            disp(ME)
            [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,C2,f2,P,options);            
        end
    else
        [C2,f2,P,S2] = update_temporal_components(Yr,A2,b2,C2,f2,P,options);
    end
    
% end


%% order ROIs
%{
fprintf('Ordering ROIs...\n')
[A_or, C_or, S_or, P, srt] = order_ROIs(A2,C2,S2,P);    % order components
%}

A_or = A2; 
C_or = C2;
S_or = S2; 
srt = [];
clear A2 C2 S2

%% extract DF/F 

fprintf('Extracting DF/F...\n')
[C_df,Df,S_df] = extract_DF_F(Yr,[A_or,b2],[C_or;f2],S_or); % extract DF/F values (optional) % FN moved it here so C_df and S_df are also ordered.


%% do some plotting

if doPlots
    contour_threshold = 0.95;                       % amount of energy used for each component to construct contour plot
    figure;
    [Coor,json_file] = plot_contours(A_or,reshape(P.sn,d1,d2),contour_threshold,1); % contour plot of spatial footprints
    pause; 
    %savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)
    view_components(Yr,A_or,C_or,b2,f2,Cn,options);         % display all components
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

