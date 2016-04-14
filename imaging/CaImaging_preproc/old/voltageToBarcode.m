function code = voltageToBarcode(volt, shortInt, longInt, lowBound, highBound, errorChecking)
% code = voltageToBarcode(volt [, shortInt] [, longInt] [, lowBound] [, highBound] [, errorChecking])
%
% Take a voltage trace containing a single barcode and extract the code.
% You may instead supply a trace of 0's and 1's for lows and highs, as long
% as you make lowBound < 1.
%
% This function does not decode the code; for that functionality, use
% decode2of5.
%
% Errors in decoding voltage will produce negative codes. Otherwise you
% will get two rows (bars and spaces) of 1's and 2's (for short and long).
%
% DEFAULTS:
% shortInt    -- 1.5  (interval to call "short")
% longInt     -- 3.5  (interval to call "long")
% lowBound    -- 0.4  (highest voltage to consider "low")
% highBound   -- 1.5  (lowest voltage to consider "high")
% errorChecking -- true  (check for valid codes)
% 
% If error checking is true, you may get the following error codes:
% -1: code too short
% -2: bad opening guard
% -3: bad closing guard
% -4: invalid length (not a multiple of 10 excluding guards)

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

if ~exist('errorChecking', 'var')
  errorChecking = true;
end


%% Threshold voltage

volt(volt < lowBound) = 0;
volt(volt >= highBound) = 1;


%% Find transitions, turn into intervals

transitionTimes = find(diff(volt));
intervals = diff(transitionTimes);


%% Turn intervals into code

code = intervals;
code(intervals <= shortInt + 1) = 1;
code(intervals >= longInt - 1) = 2;


%% Basic checks on code

if errorChecking == 1
  % Check for code too short
  if length(code) < 8
    code = -1;
    return;
  end
  
  % Check opening guard
  if ~isequal(code(1:4), [1 1 1 1])
    code = -2;
    return;
  end
  
  % Check closing guard
  if ~isequal(code(end-2:end), [1 1 1])
    code = -3;
    return;
  end
  
  % Check length
  if mod(length(code) - 7, 10) ~= 0
    code = -4;
    return;
  end
end
  
%% Convert code to 2 rows, using a trailing space

code = [code 1];

% If code isn't of even length (only possible if error checking is
% disabled), assume front was damaged and clip an element
if mod(size(code, 2), 2) ~= 0
  code = code(2:end);
end

code = reshape(code, 2, length(code) / 2);
