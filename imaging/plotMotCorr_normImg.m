function plotMotCorr_normImg(mouse, imagingFolder, mdfFileNumber)
% Assess motion correction. Also see how normalizing the movie worked (in
% order to make pixel intensities uniform before running CNMF).
%
% The very first script to run right after preproc
%
% Right after you are done with preproc on the cluster, run the following scripts:
% - plotMotionCorrOuts
% - plotEftyVarsMean (if needed follow by setPmtOffFrames to set pmtOffFrames and by findTrsWithMissingFrames to set frame-dropped trials. In this latter case you will need to rerun CNMF!): for a quick evaluation of the traces and spotting any potential frame drops, etc
% - eval_comp_main on python (to save outputs of Andrea's evaluation of components in a mat file named more_pnevFile)
% - set_mask_CC
% - findBadROIs
% - inhibit_excit_prep
% - imaging_prep_analysis (calls set_aligned_traces... you will need its outputs)

%{
mouse = 'fni19';
imagingFolder = '150930';
mdfFileNumber = [1];
%}


%% Load imfilename

signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
disp(pnev_n)
cd(fileparts(imfilename))


%% Look at motion correction outcome

load(imfilename, 'pmtOffFrames', 'outputsDFT', 'badFrames')

figure; 
% subplot(211), 
hold on;
plot(outputsDFT{1}(:,3:4))
yax = get(gca,'ylim');
plot(badFrames{1}*range(yax))

% subplot(212),
plot(pmtOffFrames{1}*range(yax))

xlabel('Frame')
ylabel('Pixel shift')
set(gcf, 'name', ['+X : brain moved left(lateral) ;+Y : brain moved up(posterior bc of 2p I think)'])
legend('Y (row) shift', 'X (column) shift', 'badFrames', 'pmtOffFrames')

pof = cellfun(@sum, pmtOffFrames);
bf = cellfun(@sum, badFrames);
t1 = sprintf('%d pmtOffFrames', pof(1));
t2 =  sprintf('%d badFrames', bf(1));
title({t1,t2})


%% Look at representative raw and motion-corrected movies

a = whos('-file', imfilename);

if any(ismember({a.name}, 'movieRawRep'))
    load(imfilename, 'movieRawRep', 'movieMCMRep')

    f = figure('position', [406         138        1244         802]); %[41         106        1244         802]);
    doimadj = 1;
    figPar.num = f.Number;
    for fr = 1:size(movieRawRep{1},3)
        figPar.subplot = [221];
        movie_play(movieRawRep{1}, fr, .02, doimadj, figPar)
        title('Raw-ch1')
        figPar.subplot = [223];
        movie_play(movieMCMRep{1}, fr, .02, doimadj, figPar)
        title('MCM-ch1')

        figPar.subplot = [222];
        movie_play(movieRawRep{2}, fr, .02, doimadj, figPar)
        title('Raw-ch2')
        figPar.subplot = [224];
        movie_play(movieMCMRep{2}, fr, .02, doimadj, figPar)
        title('MCM-ch2')
        pause(.1)
    end
    
end


%% Assess the normalization of movie that is being done before finding ROIs to make pixel intensities uniform.

load(imfilename, 'medImage', 'params')
[normingImg, softConst] = brightenFilter2DGauss(medImage{2},params);
figure('position', [25   129   553   790]); 
subplot(211), imagesc(medImage{2}), colorbar, title('medImage\_ch2')
subplot(212), imagesc(medImage{2}./(normingImg)), colorbar
title(sprintf('%s%d', 'Normalized medImage\_ch2; softConst= ', softConst))


