%% Set inhibit and excit traces.

mouse = 'fni17';
imagingFolder = '151102'; %'151029'; %  '150916'; % '151021';
mdfFileNumber = [1,2];  % 3; %1; % or tif major


saveInhibitRois = 0;
assessInhibitClass = 0; % you will go through inhibit and excit ROIs one by one.
keyEval = 0; % if 1, you can change class of neurons using key presses.
manThSet = 0; % if 1, you will set the threshold for identifying inhibit neurons by looking at the roi2surr_Sig values.
% If manThSet and keyEval are both 0, automatic identification occurs based on:
% 'inhibit: > .9th quantile. excit: < .8th quantile of roi2surr_sig
% Be very carful if setting keyEval to 1: Linux hangs with getKey if you click anywhere, just use the keyboard keys! % if 0 you will simply go though ROIs one by one, otherwise it will go to getKey and you will be able to change neural classification.


signalCh = 2; % because you get A from channel 2, I think this should be always 2.
pnev2load = [];
[imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
[pd, pnev_n] = fileparts(pnevFileName);
disp(pnev_n)
cd(fileparts(imfilename))


%% Load inhibitRois if it already exists, otherwise set it.

%     [pd,pnev_n] = fileparts(pnevFileName);
fname = fullfile(pd, sprintf('more_%s.mat', pnev_n)); % This file must be already created in python (when running Andrea's evaluate_comp code).
a = matfile(fname);

% If assessInhibit is 0 and the inhibitROI vars are already saved, we will load the results and wont perform inhibit identification anymore.
if ~assessInhibitClass && exist(fname, 'file') && isprop(a, 'inhibitRois')
    fprintf('Loading inhibitRois...\n')    
    load(imfilename, 'inhibitRois')
    
else
    fprintf('Identifying inhibitory neurons....\n')
    % inhibitRois will be :
    % 1 for inhibit ROIs.
    % 0 for excit ROIs.
    % nan for ROIs that could not be classified as inhibit or excit.
    
    [inhibitRois, roi2surr_sig, sigTh_IE] = inhibit_excit_setVars(imfilename, pnevFileName, manThSet, assessInhibitClass, keyEval);
    
    if saveInhibitRois
        fprintf('Appending inhibitRois to more_pnevFile...\n')        
        save(fname, '-append', 'inhibitRois', 'roi2surr_sig', 'sigTh_IE')
        %             save(imfilename, '-append', 'inhibitRois', 'roi2surr_sig', 'sigTh')
    end
end

fprintf('Fract inhibit %.3f, excit %.3f, unknown %.3f\n', [...
    mean(inhibitRois==1), mean(inhibitRois==0), mean(isnan(inhibitRois))])


