function [numbers, codeStarts] = segmentVoltageAndReadBarcodes(volt, shortInt, longInt, lowBound, highBound)
% [numbers, codeStarts] = segmentVoltageAndReadBarcodes(volt [, shortInt] [, longInt] [, lowBound] [, highBound])
%
% Takes a voltage trace from the analog in on the MOM, read using
% readMOMAnalog(), identifies the barcoded trial numbers, and decodes
% these into the numbers the encode.
% Remember Volt should only include the analog data related to trial codes (not the
% rest of the analog channels) (Farz).
%
% Bad codes in the voltage trace will produce negative values in numbers.
% This should not derail the decoding process, though.
%
% INPUTS
%   volt      -- the voltage trace
%   shortInt  -- the interval indicating a "short" bar or space (default 1.5)
%   longInt   -- the interval indicating a "long" bar or space (default 3.5)
%   lowBound  -- the highest voltage that should indicate a space (default 0.4)
%   highBound -- the lowest voltage that should indicate a bar (default 1.5)
%
% OUTPUTS
%   numbers   -- the decoded trial numbers
%   codeStarts -- the sample that each code started, one per "number"
%
% If any voltages are between lowBound and highBound, you will get a
% warning and these values will be called zeros.
%
% See: encode2of5, readMOMAnalog
%
% For testing, produce a voltage trace using:
% volt = []; for code = 101:110, volt = [volt generateFakeVoltageFrom2of5(encode2of5(code))]; end;


%% Parameters

minIntervalFactor = 3;


%% Optional arguments

if ~exist('shortInt', 'var')
  shortInt = 1.5;
end

if ~exist('longInt', 'var')
  longInt = 3.5;
end

if ~exist('lowBound', 'var')
  lowBound = 0.4;
end

if ~exist('highBound', 'var')
  highBound = 1.5;
end


%% Check beginning of trace, clip initial highs

if size(volt, 1) > size(volt, 2)
  volt = volt';
end

firstLow = find(volt < lowBound, 1);
if firstLow > 1
  volt = volt(firstLow:end);
end


%% Find segments

% Handle intermediate values
intermeds = find(volt >= lowBound & volt <= highBound);
if ~isempty(intermeds)
  fprintf('%d intermediate voltages found. Calling these zeros.\n', length(intermeds));
  volt(intermeds) = 0;
end

% Threshold voltage
volt(volt < lowBound) = 0;
volt(volt > highBound) = 1;

% Find transitions, turn into intervals
transitionTimes = find(diff(volt)); % Farz: this shows the time points after which the signal changes (from 0 to 1, or, 1 to 0), so it shows the index of 1st element of each barcode minus 1 and the index of the last element of each barcode.
intervals = diff(transitionTimes);

% Find overly-long intervals (last transition before break)
breakIntervals = find(intervals > minIntervalFactor * longInt); % Farz: this shows the index of the last element of each barcode. transitionTimes(breakIntervals) will show the index of the last element of barcodes in the volt trace.

nSegments = length(breakIntervals) + 1; % Farz: number of trials.

if nSegments == 1
  warning('No transitions found in voltage trace');
end


%% Feed each segment to decoder

numbers = zeros(1, nSegments);

starts = [transitionTimes(1) transitionTimes(breakIntervals+1)]; % Farz: remember transitionTimes includes the indeces of 1st and last element of all barcodes consecutively. so when transitionTimes(breakIntervals) gives the index of last element of a barcode, transitionTimes(breakIntervals+1) will give the index of 1st element of the next barcode.
ends = [transitionTimes(breakIntervals)+1 transitionTimes(end)+1];
for s = 1:nSegments
  code = voltageToBarcode(volt(starts(s):ends(s)), shortInt, longInt, lowBound, highBound);
  numbers(s) = decode2of5(code);
end


%% If there are bad codes, attempt to repair them

if any(numbers < 0)
  fprintf('Some bad codes present! Attempting to repair these.\n');
  
  % Retrieve codes without error checking
  codes = cell(1, nSegments);
  for s = 1:nSegments
    codes{s} = voltageToBarcode(volt(starts(s):ends(s)), shortInt, longInt, lowBound, highBound, 0);
  end

  % Check if there are bad codes
  nBad = sum(numbers < 0);
  
  % If there are bad codes, try to repair them
  theseRepaired = 1;
  nRepaired = 0;
  while theseRepaired > 0
    [numbers, theseRepaired] = repair2of5Sweep(numbers, codes);
    nRepaired = nRepaired + theseRepaired;
  end
  
  if nBad > 0
    fprintf('%d code(s) repaired of %d bad\n', nRepaired, nBad);
  end
end


%% Check for monotonicity in non-broken codes

goodNumbers = numbers(numbers > 0);
nNonMonotonic = sum(diff(goodNumbers) <= 0);
if nNonMonotonic > 0
  warning('%d trial codes were non-monotonic, even after excluding unrecoverable codes', ...
    nNonMonotonic);
end


%% Compute code start times

codeStarts = starts + firstLow;
