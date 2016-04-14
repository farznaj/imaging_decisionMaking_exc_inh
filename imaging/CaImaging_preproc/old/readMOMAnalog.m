function [vals, sampFreq] = readMOMAnalog(filename, headerBug)
% [vals, sampFreq] = readMOMAnalog(filename [, headerBug])
%
% Read a binary file produced by exporting the analog traces from MView.
%
% INPUTS
%   filename   -- path to the file to read
%   headerBug  -- optional. Should be true if reading a file from MView
%                 version 3.?.?.?, to correct a bug. Default false.
%
% OUTPUTS
%   vals       -- data, in volts. One row per channel.
%   sampFreq   -- sampling frequency of the data

%% Optional argument
if ~exist('headerBug', 'var')
  headerBug = 0;
end

%% Open file
[fid, message] = fopen(filename, 'rb', 'l');
if fid == -1
  error(['Unable to open file: ' filename ', error message ' message]);
end

%% Read header
% Note: apparently there's a bug in the compiler Quoc used for some
% versions of MView, which inserts a few extra bytes. This code can
% compensate.
chCount = fread(fid, 1, 'int32');
if headerBug
  fread(fid, 1, 'int32');
end
sampFreq = fread(fid, 1, 'double');
chEnabled = (fread(fid, 8, 'uint32') > 0);
chRange = fread(fid, 8, 'double');

% Skip past remainder of header, to data
if headerBug
  fseek(fid, 520, 'bof');
else
  fseek(fid, 512, 'bof');
end

%% Read data
vals = fread(fid, inf, 'int16');

%% Separate out channels
vals = reshape(vals, chCount, []);

%% Re-scale data.
% chRange appears to give the extreme plus or minus voltage value (not the
% range as the Help says), so we need to multiply by 2 to recover the
% correct voltages.
scalings = 2 * chRange(chEnabled) ./ 2^16; % Farzaneh: since the data type of vals is int16, its range is 2^16; but we want to scale it to range 20. so we multiply vals by 20/2^16 
vals = bsxfun(@times, vals, scalings);

%% Close file
fclose(fid);
