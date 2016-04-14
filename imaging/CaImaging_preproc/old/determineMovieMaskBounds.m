function [bounds, badFrames] = determineMovieMaskBounds(pixelShifts, movieRes, maxMaskWidth, showShiftHisto)
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

xShifts = pixelShifts(:, 1);
yShifts = pixelShifts(:, 2);

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

% A positive shift for x indicates the image was shifted right, so clip the
% left (low index) edge (since the algorithm we use for registration uses
% wraparound). A positive shift for y indicates the image was shifted down,
% so clip the top (low index) edge
bounds = [xShiftMax+1 movieRes(1)+xShiftMin yShiftMax+1 movieRes(2)+yShiftMin]; 


%% Figure out which frames had shifts exceeding mask bounds

% This is easy, because if there were any too-big shifts, then the mask
% will be maximum width in that direction

biggerShift = max(abs(pixelShifts), [], 2);
badFrames = (biggerShift > maxMaskWidth);

