% a very nice script (together with pnev_manual_comp_match) that allows you
% to plot and compare the trace and ROIs of 2 different methods.
%
% you can use it to compare Eftychios vs manual. Or 2 different channels.
% Or 2 different methods of Eftychios, etc.


%%
compareManual = false; % compare with manual method.
compareAnotherCh = false; % compare with ROIs and activity on a differnt channel
plotMatchedTracesAndROIs = 1;
plotOnlyMatchedTraces = 0;

mousename = 'fni17';
imagingFolder = '151102'; % '151021';
mdfFileNumber = 1; % or tif major

signalCh_meth1 = 2; % CC, mask, etc are driven from signalCh_meth1 (usually you use this as Ref, but you can change in pnev_manual_comp_match)
signalCh_meth2 = 1; % CC2, mask2, etc2 are driven from signalCh_meth2 (or manual method)

[imfilename, pnevFileName] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh_meth1);
load(imfilename, 'imHeight', 'imWidth', 'sdImage', 'medImage')

im = sdImage{signalCh_meth1}; % ROIs will be shown on im
im2 = sdImage{signalCh_meth2};

if compareManual
    gcampCh = 2;    
elseif compareAnotherCh
    [~, pnevFileName2] = setImagingAnalysisNames(mousename, imagingFolder, mdfFileNumber, signalCh_meth2);    
    disp(pnevFileName2)
else
    pnevFileName2 = input('What mat file to compare against? ');
%     pnevFileName2 = '151102_001_ch2-PnevPanResults.mat';
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Set masks and traces for comparison %%%%%%%%%%%%%%%%%%%%%%%%%%%


%% set vars for the 1st method
disp(pnevFileName)
load(pnevFileName, 'A') % load('151102_001_ch1-PnevPanResults*', 'A')
spatialComp = A; % obj.A; % A2;
clear A

if any([plotMatchedTracesAndROIs  plotOnlyMatchedTraces])
    load(pnevFileName, 'C', 'C_df', 'S_df')
    temporalComp = C;
    temporalDf = C_df;
    spikingDf = S_df;
    clear C C_df S_df
    % remove the background comp
    if size(temporalDf,1) == size(temporalComp,1)+1
        temporalDf(end,:) = [];
    end
end

% imHeight = obj.options.d1;    % P.d1;
% imWidth =  obj.options.d2;    % P.d2;


%% set vars for the 2nd method

% set spatial comp
if ~compareManual % compare with a different output from Eft algorithm.
    
    load(pnevFileName2, 'A')
    spatialComp2 = A;
    clear A
    
    %     load('151102_001_ch2-PnevPanResults.mat', 'A2', 'b2', 'C2', 'f2', 'S2')
    %     A = A2;
    %     b = b2;
    %     C = C2;
    %     f = f2;
    %     S = S2;
    %     save('151102_001_ch2-PnevPanResults.mat', '-append', 'A', 'b', 'C', 'f', 'S')
    %     rmvar('151102_001_ch2-PnevPanResults.mat', 'A2', 'b2', 'C2', 'f2', 'S2')
end



% set traces

if any([plotMatchedTracesAndROIs  plotOnlyMatchedTraces])
    if ~compareManual % compare wth a different output from Eft algorithm.
        
        load(pnevFileName2, 'C', 'C_df')
        temporalComp2 = C;
        temporalDf2 = C_df;
        clear C C_df
        % remove the background comp
        if size(temporalDf2,1) == size(temporalComp2,1)+1 
            temporalDf2(end,:) = [];
        end
        
    else % set traces for manual method.
        
        load(imfilename, 'rois', 'activity', 'pmtOffFrames')
        
        if size(activity,1)==size(temporalComp,2)
            activity = activity'; % perhaps do this for the manual activity before saving it.
        end
        activity_man = activity;
        
        % Compute df/f for the manually found activity trace.        
        smoothPts = 6;
        minPts = 7000; %800;
        dFOF_man = konnerthDeltaFOverF(activity_man, pmtOffFrames{gcampCh}, smoothPts, minPts);
        
        temporalComp2 = activity_man;
        temporalDf2 = dFOF_man;
        
    end
end


%% Set CC and mask for the 1st method

% Set ROI contours of the Spatial components found by Eftychios's algorithm.
contour_threshold = .95;
% im = sdImage{2};
[CC, ~, ~, mask] = setCC_cleanCC_plotCC_setMask(spatialComp, imHeight, imWidth, contour_threshold, im);

size(CC)
size(mask)


%% Set CC2 and mask2 for the 2nd method

if ~compareManual % compare w a different output from Eft algorithm.    
    [CC2, ~, ~, mask2] = setCC_cleanCC_plotCC_setMask(spatialComp2, imHeight, imWidth, contour_threshold, im2);    
    
else  % compare w manually-found ROIs    
    [CC2, mask2] = setCC_mask_manual(rois, im);    
end

size(CC2)
size(mask2)



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Matching ROIs and plotting traces %%%%%%%%%%%%%%%%%%%%%%%%%%%

%% you need to make sure ref and toMatch are what you want. (default var is Ref var2 is toMatch)
pnev_manual_comp_match






%%
%{
%% Plot sorted traces
% inds2plot = iPksProm;
% inds2plot = iPks(end:-1:1);
% inds2plot = iProm(end:-1:1);
inds2plot = iMeasQual(end:-1:1); % (end:-1:1);

% im = im;
manualTraceQual = 0; % if 1, a question dialogue box willb e shown that asks about traces quality.
traceQualManual = plotEftManTracesROIs(temporalDf, spikingDf, dFOF_man, spatialComp, mask, CC, rois, matchedROI_idx, im, manualTraceQual, inds2plot, plothists);


%% Plot bad-quality traces.
% badQual = find(avePks2sdS<3 | aveProm2sdS<1);
% badQual = iMeasQual(sMeasQual<0);
inds2plot = badQual;

traceQualManual = plotEftManTracesROIs(temporalDf, spikingDf, dFOF_man, spatialComp, mask, CC, rois, matchedROI_idx, im, manualTraceQual, inds2plot, plothists);
%}




%% merge ROIs
%{
thr = .8;
C_corr = corr(C');
FF1 = triu(C_corr)>= thr;                           % find graph of strongly correlated temporal components

nr = size(A,2);
A_corr = triu(A'*A);
A_corr(1:nr+1:nr^2) = 0;
FF2 = A_corr > 0;                                   % find graph of overlapping spatial components

FF3 = and(FF1,FF2);                                 % intersect the two graphs

[i, j] = ind2sub(size(FF3), find(FF3(:)>0));

figure; hold on
r = [i,j];
col = [1 0 0 ; 0 0 1];
for im=1:size(r,1)
    cnt = 0;
    h1pp = [];
    for rr = r(im,:) %1:length(CC)
        cnt = cnt+1;
        regI = 1;
        
        while regI < size(CC{rr}, 2)
            nElem = CC{rr}(1, regI);
            
            %     figure(h1)
            h1p = plot(CC{rr}(2, regI + (1:nElem)), CC{rr}(1, regI + (1:nElem)), '-', 'color', col(cnt,:));
            h1pp = [h1pp h1p];
            
            regI = regI + nElem + 1;
            %     plot(CR{roi,1}(2,:), CR{roi,1}(1,:), 'r.')
        end
        pause
    end
    delete(h1pp)
end

%}