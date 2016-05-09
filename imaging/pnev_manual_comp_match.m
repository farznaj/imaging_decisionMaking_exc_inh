% use the following to setvars for this script: pnev_manual_comp_setVars
% define vars_1 and vars_2 as either ref or toMatch

%%%%%%%%%%%%%%%%%%%%%%%%%%% Matching ROIs and plotting traces %%%%%%%%%%%%%%%%%%%%%%%%%%%

orderPlot = 'qualityOrder'; % 'consecOrder'; 'randOrder'; 'qualityOrder'; % at what order you want to plot the traces

%% Set ref and toMatch masks
% inpolygon methods
% matchROIs_inpolygon

%%% sumMask method
refMask = mask; % mask_eft;
toMatchMask = mask2; % mask_manual;



%%% Set ref and toMatch traces.
% plotMatchedTracesAndROIs = 1;
% plotOnlyMatchedTraces = 0;

refTrace_df = temporalDf; % temporalDf;
toMatchTrace_df = temporalDf2; % dFOF_man;

CC_ref = CC; % CC;
CC_toMatch = CC2; % CC_rois;

refSpatialComp = spatialComp; % spatialComp;

refSpikingDf = spikingDf; % spikingDf;

refTrace = temporalComp;
toMatchTrace = temporalComp2; % activity_man;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Measure of trace quality %%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(orderPlot, 'qualityOrder')
    %% Compute measures of trace quality
    [avePks2sdS, aveProm2sdS, measQual] = traceQualMeasure(temporalDf, spikingDf);
    
    
    %% Sort traces
    [sPks, iPks] = sort(avePks2sdS);
    [sProm, iProm] = sort(aveProm2sdS);
    [sPksProm, iPksProm] = sort(avePks2sdS .* aveProm2sdS);
    [sMeasQual, iMeasQual] = sort(measQual);
    
    
    %% Set bad-quality traces.
    % badQual = find(avePks2sdS<3 | aveProm2sdS<1);
    badQual = iMeasQual(sMeasQual<0);
    
end


%% set inds2plot
switch orderPlot
    case 'consecOrder'
        inds2plot = 1:length(CC_ref);
        
    case 'randOrder'
        inds2plot = randperm(length(CC_ref));
        
    case 'qualityOrder'        
        % inds2plot = iPksProm;
        % inds2plot = iPks(end:-1:1);
        % inds2plot = iProm(end:-1:1);
        inds2plot = iMeasQual(end:-1:1); % (end:-1:1);               
        
%         inds2plot = badQual;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% MATCHING ROIs %%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Find matching ROIs

% inpolygon methods
% matchROIs_inpolygon

% sumMask method
% matchedROI_idx(i)=j means ROI j of toMatchMask matched ROI i of refMask. % matchedROI_idx shows you what ROI from Eft method matched the manual ROI.
% matchedROI_idx(i)=NaN means no ROI in toMatchMask matched the ref mask.
if ~exist('activity_man_eftMask', 'var')
    matchedROI_idx = matchROIs_sumMask(refMask, toMatchMask); 
end

figure; plot(matchedROI_idx), % xlabel('Manual ROI index'), ylabel('matching Eft ROI index')

% compare the 2 methods
% matchROIs_inpolygon_sumMask_comp

fractOfRefWithMatch = mean(~isnan(matchedROI_idx)); % percentage of ROIs in ref mask that had a match in toMatchMask
fprintf('%.2f = Fraction of Ref RIOs with a match in the other mask.\n', fractOfRefWithMatch)


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% compare activity (manual) and temporalComp (Eft method) and their df/f versions %%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot ROI maks, contours and traces found by Eft and manual methods.

if plotMatchedTracesAndROIs    
    % CC_ref = CC;
    % CC_matched = CC2; % CC_rois;
    % inds2plot = randperm(length(CC));
    plothists = 0;
    manualTraceQual = 0; % if 1, a question dialogue box willb e shown that asks about traces quality.
    
    traceQualManual = plotEftManTracesROIs(refTrace_df, refSpikingDf, toMatchTrace_df, ...
        refSpatialComp, refMask, CC_ref, CC_toMatch, matchedROI_idx, im, refTrace, inds2plot, manualTraceQual, plothists, im2);
    
end


%% Compare Eft and manual traces, also for each Eft ROI, superimpose the corresponding manual ROI.

if plotOnlyMatchedTraces
    % refTrace = temporalComp;
    % toMatchTrace = temporalComp2; % activity_man;
    
    % refTrace_df = temporalDf;
    % matchedTrace_df = temporalDf2; % dFOF_man;
    % inds2plot = randperm(length(CC_ref));
    
    plotEftManTracesROIs0(refTrace, refTrace_df, toMatchTrace, toMatchTrace_df, matchedROI_idx, CC, rois, inds2plot)
    
end






%%
%{
traceQualManual = matchAndPlot_masksTraces(refMask, toMatchMask, plotMatchedTracesAndROIs, plotOnlyMatchedTraces, ...
    inds2plot, CC_ref, CC_toMatch, refTrace_df, toMatchTrace_df, refSpatialComp, refSpikingDf, im, ...
    refTrace, toMatchTrace);
%}

