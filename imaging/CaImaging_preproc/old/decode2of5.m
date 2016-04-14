function n = decode2of5(code)
% n = decode2of5(code)
%
% Decode the modified 2 of 5 barcode produced by encode2of5. That is, turn
% a barcode back into a number.


%% Error checking

% A single negative value for code means a bad voltage read
if numel(code) == 1 && code < 0
  n = code;
  return;
end

if size(code, 1) ~= 2
  error('Supply code as two rows, bars and spaces');
end

if mod((size(code, 2) - 4), 5) ~= 0
  error('Code is wrong length; possibly code without guards was input');
end

if ~isequal(code(:, 1:2), [1 1; 1 1])
  error('Bad opening guard');
end

if ~isequal(code(:, end-1:end), [1 1; 1 1])
  error('Bad closing guard');
end



%% Trim off guards

code = code(:, 3:end-2);


%% Reshape code to a 5 x nDigits matrix

% Transpose and reshape to get all the bars then all the spaces
code = reshape(code', 5, []);
% Cut and stack the matrices to alternate bar-digits and space-digits
nDigits = size(code, 2);
code = [code(:, 1:nDigits/2); code(:, nDigits/2 + 1:end)];
% Finally, reshape to get the digit codes, in order, as columns
code = reshape(code, 5, []);


%% Check for valid code

if ~all(sum(code) == 7)
  error('Corrupt code, each number must have 2 long modules and 3 short');
end


%% Decode digits

% These are the values for each digit in 2 of 5 barcodes
vals = [1 2 4 7 0]';

% Get values
digits = sum(bsxfun(@times, code - 1, vals), 1);
% Zeros are special, represented as 11
digits(digits == 11) = 0;


%% Convert to a single number

n = 0;
for d = 1:nDigits
  n = n + digits(d) * 10 ^ (nDigits - d);
end

