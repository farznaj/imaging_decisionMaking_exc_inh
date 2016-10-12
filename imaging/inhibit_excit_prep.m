%% Set inhibit and excit traces.

mouse = 'fni17';
imagingFolder = '151029'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [2,3];  % 3; %1; % or tif major


saveInhibitRois = 0;
assessClass_unsure_inh_excit = [0 0 0]; %[1,1,0]; % whether to assess unsure, inhibit, excit neuron classes. % you will go through unsure, inhibit and excit ROIs one by one. (if keyEval is 1, you can change class, if 0, you will only look at the images)
keyEval = 1; % if 1, you can change class of neurons using key presses.
manThSet = 0; % if 1, you will set the threshold for identifying inhibit neurons by looking at the roi2surr_Sig values.
% Be very carful if setting keyEval to 1: Linux hangs with getKey if you click anywhere, just use the keyboard keys! % if 0 you will simply go though ROIs one by one, otherwise it will go to getKey and you will be able to change neural classification.

identifInh = 1; % if 0, only bleedthrough-corrected ch1 image will be created, but no inhibitory neurons will be identified.

% Above .8 quantile is defined as inhibitory and below as excitatory.
%{
% Not doing below:
% If manThSet and keyEval are both 0, automatic identification occurs based on:
% 'inhibit: > .9th quantile. excit: < .8th quantile of roi2surr_sig
%}

%%
signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
disp(pnev_n)
cd(fileparts(imfilename))

%     [pd,pnev_n] = fileparts(pnevFileName);
moreName = fullfile(pd, sprintf('more_%s.mat', pnev_n)); % This file must be already created in python (when running Andrea's evaluate_comp code).
a = matfile(moreName);


%% Load inhibitRois if it already exists, otherwise set it.

% If assessInhibit is 0 and the inhibitROI vars are already saved, we will load the results and wont perform inhibit identification anymore.
if ~(sum(assessClass_unsure_inh_excit)) && exist(moreName, 'file') && isprop(a, 'inhibitRois')
    fprintf('Loading inhibitRois...\n')    
    load(imfilename, 'inhibitRois')
    
else
    % inhibitRois will be :
    % 1 for inhibit ROIs.
    % 0 for excit ROIs.
    % nan for ROIs that could not be classified as inhibit or excit.
    
    [inhibitRois, roi2surr_sig, sigTh_IE] = inhibit_excit_setVars(imfilename, pnevFileName, manThSet, assessClass_unsure_inh_excit, keyEval, identifInh);
    
    if saveInhibitRois
        fprintf('Appending inhibitRois to more_pnevFile...\n')        
        save(moreName, '-append', 'inhibitRois', 'roi2surr_sig', 'sigTh_IE')
        %             save(imfilename, '-append', 'inhibitRois', 'roi2surr_sig', 'sigTh')
    end
end

fprintf('Fract inhibit %.3f, excit %.3f, unknown %.3f\n', [...
    mean(inhibitRois==1), mean(inhibitRois==0), mean(isnan(inhibitRois))])


