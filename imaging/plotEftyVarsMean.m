function plotEftyVarsMean(mouse, imagingFolder, mdfFileNumber, setvars)
% If needed follow this script by codes setPmtOffFrames to set pmtOffFrames
% and by findTrsWithMissingFrames to set frame-dropped trials. In this
% latter case you will need to rerun CNMF (bc frames were dropped and
% identification of trials was wrong)!
% 
% Use this script to look at averages of ca traces (outputs of CNMF); It
% allows you to identify frame drops during a trial. Also it allows
% identifying pmtOffFrames, if there are any use the script setPmtOffFrames
% to set them and save them to imfilename.
%
% Right after you are done with preproc on the cluster, run the following scripts:
% - plotMotionCorrOuts
% - plotEftyVarsMean (if needed follow by setPmtOffFrames to set pmtOffFrames and by findTrsWithMissingFrames to set frame-dropped trials. In this latter case you will need to rerun CNMF!): for a quick evaluation of the traces and spotting any potential frame drops, etc
% - eval_comp_main on python (to save outputs of Andrea's evaluation of components in a mat file named more_pnevFile)
% - set_mask_CC
% - findBadROIs
% - inhibit_excit_prep
% - imaging_prep_analysis (calls set_aligned_traces... you will need its outputs)
%
% Required inputs:
%{
mouse = 'fni19';
imagingFolder = '150930';
mdfFileNumber = [1]; 
%}


if isnumeric(setvars) && setvars==1
    signalCh = 2; % because you get A from channel 2, I think this should be always 2.
    pnev2load = [];
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber, signalCh, pnev2load);
    [pd, pnev_n] = fileparts(pnevFileName);
    disp(pnev_n)
    cd(fileparts(imfilename))

    normalizeSpikes = 1;
    load(pnevFileName, 'activity_man_eftMask_ch2')
    load(imfilename, 'cs_frtrs')

    fprintf('Loading Eftys vars...')
    load(pnevFileName, 'C', 'C_df', 'S', 'f')
    % load(pnevFileName, 'A', 'P')
    fprintf('...done\n')

    load(imfilename, 'Nnan_nanBeg_nanEnd')
    [C, S, C_df] = processEftyOuts(C, S, C_df, Nnan_nanBeg_nanEnd, normalizeSpikes);    % S(:, [32672       32333       32439       32547]) = nan; % sharp spikes due to frame missing (their trials will be excluded... you are just doing this so they dont affect the normalization.)
else
    C = setvars{1};
    S = setvars{2};
    f = setvars{3};
    activity_man_eftMask_ch2 = setvars{4};
    cs_frtrs = setvars{5};
end


%%
cprintf('red', 'Remember about pmtOffFrames!! \n')

%
figure('position', [675         247        1351         729]); a = [];

subplot(413), hold on
top = nanmean(S);
hh = plot([cs_frtrs; cs_frtrs], [min(top); max(top)], 'g'); % mark trial beginnings
set([hh], 'handlevisibility', 'off')
if exist('nFrsSess', 'var'),
    h0 = plot([cumsum([0, nFrsSess]); cumsum([0, nFrsSess])], [min(top); max(top)], 'k'); % mark session beginnings
    h00 = plot([cs_frmovs; cs_frmovs], [min(top); max(top)], 'k:'); % mark tif movie beginnings
    set([h0; h00], 'handlevisibility', 'off');
end

plot(top); title('S'),
a = [a, gca];



subplot(412), hold on
top = nanmean(C);
hh = plot([cs_frtrs; cs_frtrs], [min(top); max(top)], 'g'); % mark trial beginnings
set([hh], 'handlevisibility', 'off')
if exist('nFrsSess', 'var'),
    h0 = plot([cumsum([0, nFrsSess]); cumsum([0, nFrsSess])], [min(top); max(top)], 'k'); % mark session beginnings
    h00 = plot([cs_frmovs; cs_frmovs], [min(top); max(top)], 'k:'); % mark tif movie beginnings
    set([h0; h00], 'handlevisibility', 'off');
end

plot(top); title('C'),
a = [a, gca];



subplot(414), hold on
top = f;
hh = plot([cs_frtrs; cs_frtrs], [min(top); max(top)], 'g'); % mark trial beginnings
set([hh], 'handlevisibility', 'off')
if exist('nFrsSess', 'var'),
    h0 = plot([cumsum([0, nFrsSess]); cumsum([0, nFrsSess])], [min(top); max(top)], 'k'); % mark session beginnings
    h00 = plot([cs_frmovs; cs_frmovs], [min(top); max(top)], 'k:'); % mark tif movie beginnings
    set([h0; h00], 'handlevisibility', 'off');
end

plot(top); title('f'),
a = [a, gca];


if exist('activity_man_eftMask_ch2', 'var')
    subplot(411), hold on
    top = nanmean(activity_man_eftMask_ch2');
    hh = plot([cs_frtrs; cs_frtrs], [min(top); max(top)], 'g'); % mark trial beginnings
    set([hh], 'handlevisibility', 'off')
    if exist('nFrsSess', 'var'),
        h0 = plot([cumsum([0, nFrsSess]); cumsum([0, nFrsSess])], [min(top); max(top)], 'k'); % mark session beginnings
        h00 = plot([cs_frmovs; cs_frmovs], [min(top); max(top)], 'k:'); % mark tif movie beginnings
        set([h0; h00], 'handlevisibility', 'off');
    end

    plot(top); title('manual, Any pmtOffFrames?!'),
    a = [a, gca];
end

linkaxes(a, 'x')
xlim([0 size(C,2)])

pause
r2 = 0;
% figure(h)
for rr = 1:floor(length(C)/.1e4)+1
    r1 = r2;
    r2 = r1+.1e4;
    xlim([r1 r2])
    %     ginput
    pause
end
    
%{
mkdir(fullfile(pd, 'figs')) % save the following 3 figures in a folder named "figs"
savefig(fullfile(pd, 'figs','caTraces_aveAllNeurons'))  
%}


%%
%{
% trace = mean(C_df); trace = f; figure; hold on; plot(trace), plot([cs_frtrs ; cs_frtrs], [min(trace) max(trace)], 'g')

xlim([0  size(C, 2)])
x = get(gca,'xlim'); len = x(end);
r2 = 0;
for rr = 1:floor(len/.5e4)+1
    r1 = r2;
    r2 = r1+.5e4;
    xlim([r1 r2])
    %     ginput
    pause
end

% clearvars -except mouse imagingFolder mdfFileNumber

%}

