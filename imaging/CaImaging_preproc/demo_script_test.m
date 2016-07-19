% Use this script for running Eftychios's algorithm on a small dataset to
% test things.

%%
nam = 'demoMovie.tif';          % insert path to tiff stack here
sframe=1;						% user input: first frame to read (optional, default 1)
num2read=2000;					% user input: how many frames to read   (optional, default until the end)
Y = bigread2(nam,sframe,num2read);

%{
mousename = 'fni17';
imagingFolder = '151102';
mdfFileNumber = 1; % or tif major
ch2read = 2;
  
tifList = tifListSet(mousename, imagingFolder, mdfFileNumber, ch2read);
tifList = tifList(end)

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


%%
K = 30;                                           % number of components to be found
tau = 4;
% order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
merge_thr = 0.8;

p=0;

options = CNMFSetParms(...
'd1',d1,'d2',d2,...                         % dimensions of datasets
'search_method','ellipse','dist',3,...      % search locations when updating spatial components
'deconv_method','constrained_foopsi',...    % activity deconvolution method
'temporal_iter',2,...                       % number of block-coordinate descent steps
'fudge_factor',0.98,...                     % bias correction for AR coefficients
'merge_thr',merge_thr,...                    % merging threshold
'gSig',tau...
);

pnev_inputParams.limit_threads = 0;
pnev_inputParams.poolsize = 0;
pnev_inputParams.maxFrsForMinPsn=[];
pnev_inputParams.save_merging_vars=0;
pnev_inputParams.multiTrs=1;
pnev_inputParams.doPlots=0;

options.limit_threads = pnev_inputParams.limit_threads;
options.poolsize = pnev_inputParams.poolsize;
options.maxFrsForMinPsn = pnev_inputParams.maxFrsForMinPsn;

save4debug=0;

finalRoundMCMC=0;
orderROI_extractDf=0;


%% run demo_script_modif from data pre-processing section to the end

%% Then go to the following function, set cs_frtrs and Nnan (see section below) and continue with the function.

params = pnev_inputParams;

[A, C, S, C_df, S_df, Df, srt, P] = update_tempcomps_multitrs(C, f, A, b, YrA, Yr, P, options, params);


%% set cs_frs and Nnan

% define fake cs_frtrs and Nnan (remember Nnan length has to be 2 less than
% cs_frtrs)
l = size(C,2);
cs_frtrs = [cumsum([0 20 100 400 36 398]), l];
Nnan = [100 200 56 24 128];

% On the NaN-ITI-inserted traces, find the index of ITI start and end, ie nanBeg and nanEnd 

% set cumsum of #frames for each tr after adding nans for ITIs
n_frtrs = diff(cs_frtrs);
ntrs_nan = [n_frtrs(1:end-1) + Nnan, n_frtrs(end)];
cs_frtrs_nan = [0 cumsum(ntrs_nan)];
% cs_frtrs_nan(1:end-1) + ntrs

% on the traces including nans, find the beg and end index for nans.
beg = cs_frtrs_nan(2:end-1)+1;
nanBeg = beg - Nnan;
nanEnd = beg-1;

Nnan_nanBeg_nanEnd = [Nnan ; nanBeg ; nanEnd];










