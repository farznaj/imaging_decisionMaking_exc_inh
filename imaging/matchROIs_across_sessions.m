mouse = 'fni17';
imagingFolder = '151102'; % '151029'; % '150916'; % '151021';
mdfFileNumber = [1,2]; % 1; % or tif major

signalCh = 2;
pnev2load = []; %7 %4 % what pnev file to load (index based on sort from the latest pnev vile). Set [] to load the latest one.

showResults = 1;


%%
mask_all_sess = cell(1, length(mdfFileNumber));
sdImage_all_sess = cell(1, length(mdfFileNumber));
COMs_all_sess = cell(1, length(mdfFileNumber));
CC_all_sess = cell(1, length(mdfFileNumber));

for isess = 1:length(mdfFileNumber)
    
    [imfilename, pnevFileName] = setImagingAnalysisNames(mouse, imagingFolder, mdfFileNumber(isess), signalCh, pnev2load);
    [~,f] = fileparts(pnevFileName);
    disp(f)
    cd(fileparts(imfilename))
    
    
    %% Set each session's mask
    
    load(imfilename, 'imHeight', 'imWidth', 'sdImage')
    load(pnevFileName, 'A') % pnevFileName should contain Eft results after merging-again has been performed.
    
    if showResults
        %     im = medImage{2};
        im = sdImage{2};
    else
        im = [];
    end
    
    contour_threshold = .95;
    fprintf('Setting the mask for the gcamp channel....\n')
    [CC, ~, COMs, mask] = setCC_cleanCC_plotCC_setMask(A, imHeight, imWidth, contour_threshold, im);
    title('ROIs shown on the sdImage of channel 2')
    % size(CC), %
    size(mask)
    
    
    %%
    mask_all_sess{isess} = mask;
    sdImage_all_sess{isess} = sdImage{1};
    COMs_all_sess{isess} = COMs;
    CC_all_sess{isess} = CC;
    
end

sdImage_all_sess
nRows = cellfun(@(x)max(size(x,1)), sdImage_all_sess);
nCols = cellfun(@(x)max(size(x,2)), sdImage_all_sess);


%% Figure out which sdImge (ie what mdfFile) should be the refImage for registration.
% I don't think you need this!
%{
% choose which session you want to use as the reference.
[~,imnr] = min(nRows);
[~,imnc] = min(nCols);

if length(unique(nRows)) > 1 && length(unique(nCols)) == 1
    irefsess = imnr;
elseif length(unique(nRows)) == 1 && length(unique(nCols)) > 1
    irefsess = imnc;
elseif length(unique(nRows)) > 1  &&  length(unique(nCols)) > 1
    if imnr==imnc
        irefsess = imnr;
    else
        error('figure out what session to choose as ref session!')
    end
end
irefsess

toregsess = [1, length(mdfFileNumber)];
toregsess(irefsess) = [];
toregsess
%}

%% Do registration on the toRegister sdImage (to find pixel shifts which you will then apply to the masks)

%%%%%% First pad the bottom and right edges with nans so both sdImages have the same size
if length(unique(nRows)) > 1  ||  length(unique(nCols)) > 1
    
    sdImage_all_sess_new = cell(1, length(mdfFileNumber));
    
    for isess = 1:length(sdImage_all_sess)
        sd_new = zeros(max(nRows), max(nCols));
        sd_new(1: nRows(isess), 1: nCols(isess)) = sdImage_all_sess{isess};
        
        sdImage_all_sess_new{isess} = sd_new;
    end
    
else
    sdImage_all_sess_new = sdImage_all_sess;
end


%%%%%% Do registration
irefsess = 1;
toregsess = 2;
usFac = 10; % upsampling factor for subpixel motion correction.

refImage = sdImage_all_sess_new{irefsess};
toRegImage = sdImage_all_sess_new{toregsess};

[dftOutputs, Greg] = dftregistration(fft2(refImage), fft2(toRegImage), usFac);
regImage = abs(ifft2(Greg));
%{
toRegImage = cat(3, refImage, sdImage_all_sess_new{toregsess});
for f = 1:size(toRegImage,3)
    [dftOutputs(f,:), Greg] = dftregistration(fft2(refImage), fft2(toRegImage(:,:,f)), usFac);
    % dftOutputs
    regImage(:,:,f) = abs(ifft2(Greg));
end
%}
sdImage_all_sess_reg = cell(1, length(mdfFileNumber));
sdImage_all_sess_reg{irefsess} = refImage; % regImage(:,:,1);
sdImage_all_sess_reg{toregsess} = regImage; % regImage(:,:,2);


%%%%%% Set maskBounds
maxMaskWidth = 20;
% maskBounds shows the bounds (left, right, top and bottom) of motion-corrected movie relative to
% the original, border-trimmed movie (size 512x402).
[maskBounds, badFrames] = determineMovieMaskBounds(dftOutputs(:, 3:4), [size(refImage,2)  size(refImage,1)], maxMaskWidth); % maskBounds shows bad pixels on left, right, top and bottom, respectively; so it corresponds to columns and rows, respectively.
maskBounds


% evaluate the results of registration
for isess = 1:length(mdfFileNumber)
    figure;
    subplot(211), imagesc(sdImage_all_sess_new{isess})
    subplot(212),  imagesc(sdImage_all_sess_reg{isess})
end



%% You don't need to do the following unless you want to check the results:
% Apply maskBounds to the registered sdImages (sdImage_all_sess_reg)

for isess = 1:length(mdfFileNumber)
    sdImage_all_sess_reg{isess} = maskMovie(sdImage_all_sess_reg{isess}, maskBounds);
end

% evaluate the results of registration
for isess = 1:length(mdfFileNumber)
    figure;
    subplot(211), imagesc(sdImage_all_sess_new{isess})
    subplot(212),  imagesc(sdImage_all_sess_reg{isess})
end


%% Register mask_all_sess using the same dftOutputs computed on sdImages

%     output=[error,diffphase,row_shift,col_shift];
diffphase = dftOutputs(2);
row_shift = dftOutputs(3);
col_shift = dftOutputs(4);

mask_all_sess_reg = cell(1, length(mdfFileNumber));

for isess = 1:length(mdfFileNumber)
    
    movie = mask_all_sess{isess};
    
    for f = 1:size(movie, 3)
        
        % Display progress
        if mod(f, 100) == 0
            fprintf('%d ', f);
        end
        if mod(f, 1000) == 0
            fprintf('\n');
        end
        
        %% Apply pixel shifts
        
        buf2ft = fft2(movie(:, :, f));
        
        [nr,nc] = size(buf2ft);
        Nr = ifftshift([-fix(nr/2):ceil(nr/2)-1]);
        Nc = ifftshift([-fix(nc/2):ceil(nc/2)-1]);
        [Nc,Nr] = meshgrid(Nc,Nr);
        Greg = buf2ft.*exp(1i*2*pi*(-row_shift*Nr/nr-col_shift*Nc/nc));
        Greg = Greg*exp(1i*diffphase);
        
        
        %%
        mask_all_sess_reg{isess}(:, :, f) = abs(ifft2(Greg)); %  uint16(abs(ifft2(Greg)))
        
        
    end
    fprintf('\n');
end


%% Set to 1 values > .5 in the registered masks and to 0 values < .5.

for isess = 1:length(mdfFileNumber)
    mask_all_sess_reg{isess}(mask_all_sess_reg{isess}>.5) = 1;
    mask_all_sess_reg{isess}(mask_all_sess_reg{isess}<.5) = 0;
end


%% Apply maskBounds to mask_all_sess

for isess = 1:length(mdfFileNumber)
    mask_all_sess_reg{isess} = maskMovie(mask_all_sess_reg{isess}, maskBounds);
end
mask_all_sess_reg
% figure; imagesc(mask_all_sess_reg{1}(:,:,1))
% figure; imagesc(mask_all_sess_reg{2}(:,:,1))


%% Now match ROIs between the two masks

refMask = mask_all_sess_reg{1};
toMatchMask = mask_all_sess_reg{2};

matchedROI_idx = matchROIs_sumMask(refMask, toMatchMask);
% matchedROI_idx : shows which toMatch ROI had max overlap with each ref ROI.
% matchedROI_idx(i) = j means ROI j of toMatchMask matched ROI i of refMask.
% matchedROI_idx(i) = NaN means no ROI in toMatchMask matched the ref mask.

figure; plot(matchedROI_idx), xlabel('Ref ROI index'), ylabel('toMatch ROI index')


%% Assess the results of matching

figure;
for i = 221; % randperm(size(refMask,3))
    set(gcf,'name',num2str(i))
    subplot(131), imagesc(refMask(:,:,i))
    subplot(132), imagesc(toMatchMask(:,:,matchedROI_idx(i)))
    subplot(133), imagesc(refMask(:,:,i) + toMatchMask(:,:,matchedROI_idx(i)))
    drawnow, pause
end



%% you are working on an alternative faster method for matching ROIs, once
% done go back


%% Correct COMs for pixel shifts (ie apply maskBounds to COMs)

COMs_all_sess_new = COMs_all_sess;

for isess = 1:length(mdfFileNumber)
    % y
    COMs_all_sess_new{isess}(:,1) = COMs_all_sess_new{isess}(:,1) - (maskBounds(1,3)-1);
    % x
    COMs_all_sess_new{isess}(:,2) = COMs_all_sess_new{isess}(:,2) - (maskBounds(1,1)-1);
end


%% A faster way than matchROIs_sumMask for finding matching ROIs

% diffCOMs = NaN(1, size(COMs_all_sess_new{1},1));
imatch = NaN(1, size(COMs_all_sess_new{1},1));
fractOverlap = NaN(2, size(COMs_all_sess_new{1},1));

imatch = cell(1, size(COMs_all_sess_new{1},1));
imatch(:) = {NaN};
fractOverlap = cell(1, size(COMs_all_sess_new{1},1));
fractOverlap(:) = {NaN};

maskOverlapMeasure = NaN(size(toMatchMask,3), size(refMask,3));

for iref = 1:size(COMs_all_sess_new{1},1)
    b = abs(bsxfun(@minus, COMs_all_sess_new{1}(iref,:), COMs_all_sess_new{1}));
    bb = sum(abs(b), 2);
    %     [diffCOMs(iref), imatch(iref)] = min(bb); % which ROI in mask{2} matched ROI iref in mask{1}
    
    % For ROIs that are near each other use the sum_mask method to find the
    % best matching one.
    f = find(bb<10)'; % this threshold is important
    if ~isempty(f)
%         clear maskOverlapMeasure
        %     maskOverlapMeasure = NaN(size(toMatchMask,3), 1);
        for itoMatch = f
            sumMask = sparse(toMatchMask(:,:,itoMatch)) + sparse(refMask(:,:,iref));
            maskOverlapMeasure(itoMatch, iref) = sum(sumMask(:)>1);
        end
        mom = maskOverlapMeasure(:, iref);
%         mom(mom~=0)
%         mom(~isnan(mom))
        a = refMask(:,:,iref);
        %{      
        [m, imatch(iref)] = max(mom);
        
        % compute fraction of overlap (fraction of iref and itomatch rois
        that are overlapping), so later you exclude matched ROIs % that
        overlap only a tiny bit! aa = toMatchMask(:,:,imatch(iref));
        fractOverlap(:, iref) = full([m / sum(a(:))  m / sum(aa(:))]);
        %}        
        m = max(mom);
        imatch{iref} = find(mom==max(mom)); % in case there are multiple max values. % find ROIs that have max overlap.
        
        fractOverlap{iref} = m / sum(a(:));
        for ii = 1:length(imatch{iref})
            aa = toMatchMask(:,:,imatch{iref}(ii));
            fractOverlap{iref} = [fractOverlap{iref}, m / sum(aa(:))];
        end
        fractOverlap{iref} = full(fractOverlap{iref});
    end
    
end

fractOverlapMean = cell(1, size(COMs_all_sess_new{1},1));
fractOverlapMean(:) = {NaN};
for iref = 1:size(COMs_all_sess_new{1},1)
    fractOverlapMean{iref} = arrayfun(@(x)mean([fractOverlap{iref}(1), x]), fractOverlap{iref}(2:end));
end

figure; plot(cellfun(@(x)sum(x<.4), fractOverlapMean))

merged_ROIs_new = imatch(cellfun(@length,imatch)>1)
find((cellfun(@length,imatch)>1))


%% exclude matched ROIs that overlap only a tiny bit!

thOverlap = .4;
imatch(mean(fractOverlap,1) < thOverlap) = NaN;

figure; plot(imatch)
% figure; plot(diffCOMs)
figure; plot(fractOverlap')
figure; plot(mean(fractOverlap,1))


% it is a much faster method.
% diffCOMs needs a thresh ... 12 seems good... it will be clear in sum of masks..


% is there a reason you bother with all of this instead of the following:
% how about you go with:  concat raw tifs of all sessions then do MC, then proceed with your multiple-sess-per-day notes.
% yes because you will need to match ROIs across sessions too... (not just
% sessions of the same day)


% what to do with ROIs that need to be merged...% only problem is when several neurons have the same matched ROI.

% add fractOverlap to matchROIs_sumMask

% use this matching analysis to figure out what ROIs need to be merged,
% then proceed with your scripts... find them... u already worked on manual
% merging!
% find(imatch==20)
% go through your merge_again codes... this is what you need to figure out now.


%% Assess the results of matching

% a = mean(fractOverlap,1);
% f = a < .4;

figure;
% cnt=0;
for iref = randperm(size(refMask,3)) % find(f); %
%     cnt=cnt+1;
    if ~isnan(imatch(iref))
        set(gcf,'name',num2str(iref))
        
        mn = COMs_all_sess_new{1}(iref,:) - 20;
        mx = COMs_all_sess_new{1}(iref,:) + 20;
        
        subplot(131), imagesc(refMask(:,:,iref))
        xlim([mn(2) mx(2)])
        ylim([mn(1) mx(1)])
        title(a(iref))
        
        subplot(132), imagesc(toMatchMask(:,:,imatch(iref)))
        xlim([mn(2) mx(2)])
        ylim([mn(1) mx(1)])
        
        subplot(133), imagesc(refMask(:,:,iref) + toMatchMask(:,:,imatch(iref)))
        xlim([mn(2) mx(2)])
        ylim([mn(1) mx(1)])
        
        drawnow, pause
    end
end
