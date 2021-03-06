function [bounds, badFrames] = determineMovieMaskBounds_fn(pixelShifts, movieRes, maxMaskWidth, showShiftHisto)
% [bounds, badFrames] = determineMovieMaskBounds(pixelShifts, movieRes, maxMaskWidth [, showShiftHisto])
%
% INPUTS
%   pixelShifts   -- output from motionCorrectCaImagingFile()
%   movieRes      -- [width height] of images
%   maxMaskWidth  -- the widest mask permitted. If there are shifts larger
%                    than this, those frames will be marked bad
%   showShiftHisto -- optional. If true, display a histogram of x shifts
%                     and a histogram of y shifts. Default false
%
% OUTPUTS
%   bounds        -- the mask bounds, to use as input to maskMovie. bounds
%                    is [x1 x2 y1 y2]
%   badFrames     -- a logical vector with size nFrames x 1. When a frame
%                    has a larger pixel shift during motion correction than
%                    the width of the mask, it is marked as bad (true).


%% Optional arguments

if ~exist('showShiftHisto', 'var')
  showShiftHisto = 0;
end


%% Find largest pixel shifts in each direction

yShifts = pixelShifts(:, 1); % row shifts % Farz changed. matt had xshifts and yshifts the other way. but dftregistration says column3 of dftoutputs is row shifts and column 4 is col shifts.
xShifts = pixelShifts(:, 2); % col shifts

% If asked, plot histograms of the pixel shifts
if showShiftHisto
  figure;
  hist(xShifts, 50);
  title('x shifts');
  
  figure;
  hist(yShifts, 50);
  title('y shifts');
end

% How much images were shifted in each direction (rounded up)
xShiftMin = floor(min(xShifts));
xShiftMax = ceil(max(xShifts));
yShiftMin = floor(min(yShifts));
yShiftMax = ceil(max(yShifts));


%% Apply maxMaskWidth limit to shifts
% That is, never mask off more then maxMaskWidth pixels on a side

xShiftMin = max([-maxMaskWidth xShiftMin]);
xShiftMax = min([maxMaskWidth xShiftMax]);
yShiftMin = max([-maxMaskWidth yShiftMin]);
yShiftMax = min([maxMaskWidth yShiftMax]);


%% Determine mask
% see this:
% http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation/content/html/efficient_subpixel_registration.html

% registered image (regMovie) is the raw image after being shifted to align the reference image.

% A positive shift for x indicates the image had to be shifted right to get
% registered (i.e to get aligned with the reference image), so clip the left (low index) edge 
% (since the algorithm we use for registration uses wraparound). 

% A positive shift for y indicates the
% image had to be shifted down to te aligned, so clip the top (low index) edge

% bounds = [xShiftMax+1 movieRes(1)+xShiftMin yShiftMax+1 movieRes(2)+yShiftMin]; % Farz commented.

bounds = [max(1, xShiftMax+1)   min(movieRes(1), movieRes(1)+xShiftMin) ...
    max(1, yShiftMax+1)   min(movieRes(2), movieRes(2)+yShiftMin)]; % Farz editted to account for motions that are always in one direction (i.e. x (or y) shift is always positive (or negative).).




%% Figure out which frames had shifts exceeding mask bounds

% This is easy, because if there were any too-big shifts, then the mask
% will be maximum width in that direction

biggerShift = max(abs(pixelShifts), [], 2);
badFrames = (biggerShift > maxMaskWidth);

