function [A_or, C_or, S_or, C_df, S_df, Df, srt, P, Nnan_nanBeg_nanEnd] = update_tempcomps_multitrs(C, f, A2, b2, YrA, Yr, P, options, params)
% Main function for estimating C and S on multi-trial traces for which the
% ITI (iter-trial interval) was not recorded. 
%
% Suggested by Eftychios Pnevmatikakis:
% Run demo_script with P.p = 0; Then use this function. 
% It estimates a single g (related to spike time constants) and sn (related
% to noise in the trace) for each neuron by taking into account all trial
% traces at the same time (as opposed to estimating g and sn from the
% concatenated trace). Then this g is used to do deconvolution on the trace
% which contains NaNs for the ITIs. ITI duration is estimated from the
% behavioral data. Also, an upper limit is used for the number of NaNs
% given our estimate of max Tau_decay of spikes (1000ms).
% 
% C_df, S_df, Df and f wont have the ITI parts. C_or and S_or have
% estimates for the ITI parts. Nnan_nanBeg_nanEnd: 1st row shows number of
% nans (frames) that were added for each ITI. 2nd row shows the index of
% ITI start on the nan-iti traces (like C_or, etc). 3rd row shows the index
% of ITI end on those traces.
%
% Farzaneh Najafi
% Apr 2016 


%% Set cs_frtrs and Nnan

cs_frtrs = params.cs_frtrs;
Nnan = params.Nnan;

%{
[~,mousename] = fileparts(fileparts(fileparts(params.tifFold)));
imagingFolder = num2str(params.tifNums(1, 1));
mdfFileNumber = params.tifNums(1, 2);
allTifMinors = params.allTifMinors;
tifMinor = unique(params.tifNums(:,3))';

[cs_frtrs, Nnan] = update_tempcomps_multitrs_setvars(mousename, imagingFolder, mdfFileNumber, allTifMinors, tifMinor);
%}


%% Perform deconvolution

fprintf('Updating of temporal components started (p=2).\n')
[C2, f2, P, S2] = deconv_multitrs(cs_frtrs, Nnan, YrA, C, f, P, options);


%% Order ROIs

fprintf('Ordering ROIs...\n')
[A_or, C_or, S_or, P, srt] = order_ROIs(A2,C2,S2,P);    % order components



%% Remove ITI parts from C, S and f to prepare them for DF extraction

% on the NaN-ITI-inserted traces, find the index of ITI start and end, ie nanBeg and nanEnd 

% set cumsum of #frames for each tr after adding nans for ITIs
n_frtrs = diff(cs_frtrs);
ntrs_nan = [n_frtrs(1:end-1) + Nnan, n_frtrs(end)];
cs_frtrs_nan = [0 cumsum(ntrs_nan)];
% cs_frtrs_nan(1:end-1) + ntrs

% on the traces including nans, find the beg and end index for nans.
beg = cs_frtrs_nan(2:end-1)+1;
nanBeg = beg - Nnan;
nanEnd = beg-1;


inds2keep = cell2mat(arrayfun(@(x,y)(x:y), [1 nanEnd+1], [nanBeg-1 size(C2,2)], 'uniformoutput', 0));

cnonan = C_or;
cnonan = cnonan(:, inds2keep);

s = S_or; % C_multi_trs;
s = s(:, inds2keep);


fnonan = f2;
fnonan = fnonan(inds2keep);

Nnan_nanBeg_nanEnd = [Nnan ; nanBeg ; nanEnd];


%% Extract DF/F

fprintf('Extracting DF/F...\n')
K_m = size(C_or, 1);
[C_df, Df, S_df] = extract_DF_F(Yr, [A_or,b2], [cnonan;fnonan], s, K_m+1); % extract DF/F values (optional) % FN moved it here so C_df and S_df are also ordered.
% [C_df, Df, S_df] = extract_DF_F(Yr, [A_or,b2], [C_or;f2], S_or, K_m+1); % extract DF/F values (optional) % FN moved it here so C_df and S_df are also ordered.


%% Do some plotting

if params.doPlots
    Cn =  reshape(P.sn, options.d1, options.d2); %correlation_image(Y); %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)
    contour_threshold = 0.95;                       % amount of energy used for each component to construct contour plot
    figure;
    [Coor,json_file] = plot_contours(A_or,reshape(P.sn, options.d1, options.d2),contour_threshold,1); % contour plot of spatial footprints
    pause;
    %savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)
    %     view_components(Yr,A_or,C_or,b2,f2,Cn,options);         % display all components
    % display components
    plot_components_GUI(Yr ,A_or, cnonan, b2, fnonan, Cn, options)
%     plot_components_GUI(Yr ,A_or, C_or, b2, f2, Cn, options)
end


%% Make movie

if params.doPlots
    make_patch_video(A_or, cnonan, b2, fnonan, Yr, Coor, options)
%     make_patch_video(A_or, C_or, b2, f2, Yr, Coor, options)
end


