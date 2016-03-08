function [A_m, C_m, S_m, C_df_m, S_df_m, Df_m, b_m, f_m, srt, Ain, options, P_m, merging_vars] = demo_script_finalMerge(Y)
% [A, C, S, C_df, S_df, Df, b, f, srt, Ain, options, P, merging_vars] = demo_script_finalMerge(movieMC{params.activityCh});

%%
if ~isa(Y,'double');    Y = double(Y);  end         % convert to double

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

Yr = reshape(Y,d,T);
clear Y

%%
mousename = 'fni17';
imagingFolder = '151102'; % '151021';
mdfFileNumber = 1; % or tif major

signalCh = 2; % CC, mask, etc are driven from signalCh_meth1 (usually you use this as Ref, but you can change in pnev_manual_comp_match)

[~, pnevFileName] = setImagingAnalysisNames(mousename, imagingFolder, ...
    mdfFileNumber, signalCh);

load(pnevFileName, 'A', 'C', 'b', 'f', 'S', 'P', 'options', 'merged_ROIs_new')


%% run Eft code to get A, C and S after merging
[Am,Cm,K_m,~,Pm,Sm] = merge_components_again(merged_ROIs_new,A,b,C,f,P,S,options);


%% repeat
%{
rrs = size(Cm,1)-length(merged_ROIs)+1 : size(Cm,1);
Cmn = Cm(rrs,:);
Amn = Am(:,rrs);
Pmn.gn = Pm.gn(rrs);
Pmn.b = Pm.b(rrs);
Pmn.c1 = Pm.c1(rrs);
Pmn.neuron_sn = Pm.neuron_sn(rrs);
%}

[A2,b2] = update_spatial_components(Yr,Cm,f,Am,Pm,options);
[C2,f2,Pm,S2] = update_temporal_components(Yr,A2,b2,Cm,f,Pm,options);
[C_df,Df,S_df] = extract_DF_F(Yr,[A2,b2],[C2;f2],S2,K_m+1); % extract DF/F values (optional)


%%
srt = [];
merging_vars = [];
Ain = [];

A_m = A2;
C_m = C2;
S_m = S2;
b_m = b2;
f_m = f2;
P_m = Pm;
C_df_m = C_df;
Df_m = Df;
S_df_m = S_df;


