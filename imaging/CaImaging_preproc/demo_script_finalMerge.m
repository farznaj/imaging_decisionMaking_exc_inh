function [A_m, C_m, S_m, C_df_m, S_df_m, Df_m, b_m, f_m, merging_vars_m, options_m, P_m] = demo_script_finalMerge(mouse, imagingFolder, mdfFileNumber)

%{
mousename = 'fni17';
imagingFolder = '151102'; % '151021';
mdfFileNumber = 1; % or tif major
%}

doUpdate = 1; % if 1, after merging updating of spatial and temporal components will be performed.


%%

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh);
[~,f] = fileparts(pnevFileName);
disp(f)


%%
load(pnevFileName, 'C_df', 'A', 'C', 'b', 'f', 'S', 'P', 'options') % , 'merged_ROIs_new')

P.p = 2; %
options.merge_thr = 0.7;
options.A_thr = 0.3;


%% Remove ITI parts from C, S

if size(C,2) ~= size(C_df,2) % iti-nans were inserted in C and S: remove them.
    load(imfilename, 'Nnan_nanBeg_nanEnd')
    nanBeg =  Nnan_nanBeg_nanEnd(2,:);
    nanEnd = Nnan_nanBeg_nanEnd(3,:);
    inds2rmv = cell2mat(arrayfun(@(x,y)(x:y), nanBeg, nanEnd, 'uniformoutput', 0)); % index of nan-ITIs (inferred ITIs) on C and S traces.
    C(:, inds2rmv) = [];
    
    if size(S,2) ~= size(C_df,2)
        S(:, inds2rmv) = [];
    end
end

clear C_df


%% Merge components

[Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components([],A,b,C,f,P,S,options);

clear A C

% run Eft code to get A, C and S after merging
% [Am,Cm,K_m,~,Pm,Sm] = merge_components_again(merged_ROIs_new,A,b,C,f,P,S,options);





%% Read the movie (Y).
% Set some initial parameters

if isunix
    dataPath = '/sonas-hs/churchland/nlsas/data/data';
elseif ispc
    dataPath = '\\sonas-hs.cshl.edu\churchland\data';
end
tifFold = fullfile(dataPath, mouse, 'imaging', imagingFolder);

% Set the tif files corresponding to mdf file mdfFileNumber and channel ch2ana
ch2ana = 2;
files = dir(fullfile(tifFold, sprintf('%s_%03d_*_ch%d_MCM.TIF', imagingFolder, mdfFileNumber, ch2ana)));
% tifList = {files.name}
tifList = cell(1, length(files));
for itif = 1:length(files)
    tifList{itif} = fullfile(tifFold, files(itif).name);
end
% showcell(tifList')


%% Read tif files into movieMC

Y = [];
for t = 1:length(tifList)
    fprintf('Reading tif file %s\n', tifList{t})
    Y = cat(3, Y, bigread2(tifList{t}));
end


%%
if ~isa(Y,'double');    Y = double(Y);  end         % convert to double

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

Yr = reshape(Y,d,T);
clear Y


%% Repeat updating of spatial and temporal components and extract df/f.

%{
rrs = size(Cm,1)-length(merged_ROIs)+1 : size(Cm,1);
Cmn = Cm(rrs,:);
Amn = Am(:,rrs);
Pmn.gn = Pm.gn(rrs);
Pmn.b = Pm.b(rrs);
Pmn.c1 = Pm.c1(rrs);
Pmn.neuron_sn = Pm.neuron_sn(rrs);
%}

if doUpdate
    fprintf('Updating spatial components...\n')
    [A2,b2,Cm] = update_spatial_components(Yr,Cm,f,Am,Pm,options);
    
    fprintf('Updating temporal components...\n')
    [C2,f2,Pm,S2] = update_temporal_components(Yr,A2,b2,Cm,f,Pm,options);
    complexTol = 1e-10;
    [A2, C2, S2, Pm] = removeComplexUnits(A2, C2, S2, f2, complexTol, Pm);
    
    fprintf('Extracting DF/F...\n')
    [C_df,Df,S_df] = extract_DF_F(Yr,[A2,b2],[C2;f2],S2,K_m+1); % extract DF/F values (optional)

    A_m = A2;
    C_m = C2;
    S_m = S2;
    b_m = b2;
    f_m = f2;

    clear A2 C2 S2 b2 f2
else
    fprintf('Extracting DF/F...\n')
    [C_df,Df,S_df] = extract_DF_F(Yr,[Am,b],[Cm;f],Sm,K_m+1); % extract DF/F values (optional)
   
    A_m = Am;
    C_m = Cm;
    S_m = Sm;
    b_m = b;
    f_m = f;
    
    clear Am Cm Sm b f
end



%%
merging_vars_m.merged_ROIs = merged_ROIs;
options_m = options;

P_m = Pm;
C_df_m = C_df;
Df_m = Df;
S_df_m = S_df;

clear C_df Df S_df


%% Save: append to pnevFileName

fprintf('Saving and appending to pnevFileName ...')
save(pnevFileName, '-append', 'A_m', 'C_m', 'S_m', 'C_df_m', 'S_df_m', 'Df_m', 'b_m', 'f_m', 'merging_vars_m', 'options_m', 'P_m')



