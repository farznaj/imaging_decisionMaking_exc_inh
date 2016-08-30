% Right after you are done with preproc on the cluster, run
% eval_comp_main on Python to save outputs of Andrea's evaluation of
% components in a mat file named more_pnevFile... . Then run this script to
% append to that matfile mask and CC.


%% Change these vars:

mouse = 'fni17';
imagingFolder = '151101'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1];  % 3; %1; % or tif major


%% Set imfilename, pnevFileName, fname

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];

[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd,pnev_n] = fileparts(pnevFileName);
disp(pnev_n)
cd(fileparts(imfilename))

fname = fullfile(pd, sprintf('more_%s.mat', pnev_n));


%% Set mask and CC

fname = fullfile(pd, sprintf('more_%s.mat', pnev_n));

load(pnevFileName, 'A')
load(imfilename, 'imHeight', 'imWidth')

mask = maskSet(A, imHeight, imWidth);
CC = ROIContoursPnevCC(A, imHeight, imWidth, .95);


%% Append A and CC to a mat file named more_pnev created in python... (you have to first create this file in python, it includes outputs of Andrea's evaluate components).

if exist(fname,'file')==2
    fprintf('Appending mask and CC to more_pnevFile....\n')
    save(fname, 'mask', 'CC', '-append')
else
    error('You should first creat this mat file in python, otherwise you wont be able to append evaluate_comps vars to it later in python!')
    save(fname, 'mask', 'CC')
end




%%
% Use below for those days that you need to get srt_val, or to normalize
% S,... in general for all days analysis.
%{
load(pnevFileName, 'A','C','S','P', 'options', 'YrA')

%%
fprintf('Ordering ROIs...\n')
t1 = tic;
[A_or, C_or, S_or, P, srt, srt_val, nA] = order_ROIs(A,C,S,P, options);    % order components
fprintf('\norder_ROIs took %0.1f s\n\n', toc(t1));
YrA = YrA(srt,:);


%% Get rid of the background component in C_df

if size(C_df,1) == size(C,1)+1
    %     bk_df = temporalDf(end,:); % background DF/F
    C_df(end,:) = [];
end

%% sort the following vars too

load(pnevFileName, 'activity_man_eftMask_ch2', 'activity_man_eftMask_ch1', 'C_df', 'Df')


%% Remove S_df if it is saved

%% Append the modified vars to pnevFile
%}