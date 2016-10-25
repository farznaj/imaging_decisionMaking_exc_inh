function inhibit_excit_prep(mouse, imagingFolder, mdfFileNumber, saveInhibitRois, assessClass_unsure_inh_excit, keyEval, manThSet, identifInh, do2dGauss)
% Identify inhibit and excit ROIs using the gcamp channel and the tdtomato channel.

%{
mouse = 'fni17';
imagingFolder = '151020'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1,2];  % 3; %1; % or tif major


saveInhibitRois = 1;
assessClass_unsure_inh_excit = [1 1 1]; %[1,1,0]; % whether to assess unsure, inhibit, excit neuron classes. % you will go through unsure, inhibit and excit ROIs one by one. (if keyEval is 1, you can change class, if 0, you will only look at the images)
keyEval = 1; % if 1, you can change class of neurons using key presses.
manThSet = 0; % if 1, you will set the threshold for identifying inhibit neurons by looking at the roi2surr_Sig values.
% Be very carful if setting keyEval to 1: Linux hangs with getKey if you click anywhere, just use the keyboard keys! % if 0 you will simply go though ROIs one by one, otherwise it will go to getKey and you will be able to change neural classification.

identifInh = 1; % if 0, only bleedthrough-corrected ch1 image will be created, but no inhibitory neurons will be identified.
do2dGauss = 0; % Do 2D gaussian fitting on ch1 images of gcamp ROIs for ROIs identified by your measure as unsure.

% Above .8 quantile is defined as inhibitory and below as excitatory.
%{
% Not doing below:
% If manThSet and keyEval are both 0, automatic identification occurs based on:
% 'inhibit: > .9th quantile. excit: < .8th quantile of roi2surr_sig
%}

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
    
    [inhibitRois, roi2surr_sig, sigTh_IE, x_all, cost_all] = inhibit_excit_setVars(imfilename, pnevFileName, manThSet, assessClass_unsure_inh_excit, keyEval, identifInh, do2dGauss);
    
    if saveInhibitRois
        fprintf('Appending inhibitRois to more_pnevFile...\n')
        if ~isempty(x_all)
            save(moreName, '-append', 'inhibitRois', 'roi2surr_sig', 'sigTh_IE', 'x_all', 'cost_all')
        else
            save(moreName, '-append', 'inhibitRois', 'roi2surr_sig', 'sigTh_IE')
        end
        %             save(imfilename, '-append', 'inhibitRois', 'roi2surr_sig', 'sigTh')
    end
end

% fprintf('Fract inhibit %.3f, excit %.3f, unknown %.3f\n', [...
%     mean(inhibitRois==1), mean(inhibitRois==0), mean(isnan(inhibitRois))])

cprintf('blue', '%d inhibitory; %d excitatory; %d unsure neurons in gcamp channel.\n', sum(inhibitRois==1), sum(inhibitRois==0), sum(isnan(inhibitRois)))
cprintf('blue', '%.1f%% inhibitory; %.1f%% excitatory; %.1f%% unsure neurons in gcamp channel.\n', mean(inhibitRois==1)*100, mean(inhibitRois==0)*100, mean(isnan(inhibitRois)*100))


