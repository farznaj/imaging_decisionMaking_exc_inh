function [C, S, C_df] = processEftyOuts(C, S, C_df, Nnan_nanBeg_nanEnd, normalizeSpikes)
% cleans C and S, ie removes the iti nans.
% removes the last row of C_df (background comp).
% if normalizeSpikes is 1, normalizes S of each comp to its max.
% P and A are optional, if provided plots of tau, neuron_sn and max(A) will be made.

% load(pnevFileName, 'C', 'C_df', 'S', 'A', 'P')
% load(imfilename, 'Nnan_nanBeg_nanEnd')
% normalizeSpikes = 1;
% frameLength = 1000/30.9; % sec.


%%

if normalizeSpikes
    fprintf('Normalizing spikes traces of each neuron.\n')
    S = bsxfun(@rdivide, S, max(S,[],2)); % normalize spikes trace of each neuron by its max.
    %     spikes = bsxfun(@rdivide, spikes, quantile(spikes,.9)); % normalize spikes trace of each neuron by its 90th percentile.
end

% if iti-nans were inserted in C and S: remove them.
if size(C,2) ~= size(C_df,2)
%     load(imfilename, 'Nnan_nanBeg_nanEnd')
    nanBeg =  Nnan_nanBeg_nanEnd(2,:);
    nanEnd = Nnan_nanBeg_nanEnd(3,:);
    inds2rmv = cell2mat(arrayfun(@(x,y)(x:y), nanBeg, nanEnd, 'uniformoutput', 0)); % index of nan-ITIs (inferred ITIs) on C and S traces.
    C(:, inds2rmv) = [];
    
    if size(S,2) ~= size(C_df,2)
        S(:, inds2rmv) = [];
    end
end

% get rid of the background component in C_df
if size(C_df,1) == size(C,1)+1
    %     bk_df = temporalDf(end,:); % background DF/F
    C_df(end,:) = [];
end


% check if there are any nan traces.
an = find(~sum(~isnan(C),2));
if ~isempty(an)
    warning(['C trace of neuron(s) ', repmat('%i ', 1, length(an)), 'is all NaN. This should not happen!'], an);
end
an = find(~sum(~isnan(C_df),2));
if ~isempty(an)
    warning(['C_df trace of neuron(s) ', repmat('%i ', 1, length(an)), 'is all NaN. This should not happen!'], an);
end


% Use below if you want to look at the backgound component, aligned on different trial events.:
%{
load(pnevFileName, 'f')
dFOF = f';
spikes = dFOF;
%}

% Use below if you want to look at manual DF/F:
%{
% load(imfilename, 'activity_custom') % dark ROIs
% activity = activity_custom{2};
load(pnevFileName, 'activity_man_eftMask') % manual activity of Efty's ROIs
activity = activity_man_eftMask;
gcampCh = 2; smoothPts = 6; minPts = 7000; %800;
dFOF = konnerthDeltaFOverF(activity, pmtOffFrames{gcampCh}, smoothPts, minPts);
spikes = dFOF;
%}



% Assess the shape of merged ROIs:
%{
% before ordering ROIs, the last components in A and C are the merged
% components. So srt==lastComps will give the index (in the ordered ROI
% array) that corresponds to lastComps in the original A, ie the
% mergedComps.
load(pnevFileName, 'srt', 'merging_vars')
figure; plot(srt)
figure;
for i = size(A,2):-1: size(A,2)-length(merging_vars.merged_ROIs)+1 % size(A,2):-1:1
    imagesc(reshape(A(:,srt==i), imHeight, imWidth))
    pause
end
%}

%{
figure;
for i = size(A,2):-1:1 % 0:(length(merging_vars.merged_ROIs)-1)
    imagesc(reshape(merging_vars.Am(:,i), imHeight, imWidth)) % (:,end-i)
    pause
end
%}
