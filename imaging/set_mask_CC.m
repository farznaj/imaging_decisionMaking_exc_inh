function set_mask_CC(mouse, imagingFolder, mdfFileNumber)
% Right after you are done with preproc on the cluster, run the following scripts:
% - plotEftyVarsMean (if needed follow by setPmtOffFrames to set pmtOffFrames and by findTrsWithMissingFrames to set frame-dropped trials. In this latter case you will need to rerun CNMF!): for a quick evaluation of the traces and spotting any potential frame drops, etc
% - eval_comp_main on python (to save outputs of Andrea's evaluation of components in a mat file named more_pnevFile)
% - set_mask_CC
% - findBadROIs
% - inhibit_excit_prep
% - imaging_prep_analysis (calls set_aligned_traces... you will need its outputs)
%
% Example inputs:
%{
mouse = 'fni17';
imagingFolder = '151021'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1];  % 3; %1; % or tif major
%}


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


fprintf('Done!\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot ROIs found by Eftychios's algorithm on the sdImage

load(imfilename, 'sdImage')
im = sdImage{2};
% im = normImage(sdImage{2});
plotCOMs = 0;

if exist('im', 'var') && ~isempty(im)
    colors = hot(2*size(CC,1));
    colors = colors(end:-1:1,:);
    
    figure;
    imagesc(im);  % imagesc(log(im));
    hold on;
%     colormap gray
    
    for rr = 1:length(CC) % find(~badROIs01)' 
        if plotCOMs
            COMs = fastCOMsA(A, [imHeight, imWidth]);
            plot(COMs(rr,2), COMs(rr,1), 'r.')
            
        else
            %[CC, ~, COMs] = setCC_cleanCC_plotCC_setMask(Ain, imHeight, imWidth, contour_threshold, im);
            if ~isempty(CC{rr})
                plot(CC{rr}(2,:), CC{rr}(1,:), 'color', colors(rr, :))
            else
                fprintf('Contour of ROI %i is empty!\n', rr)
            end
        end
    end
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
