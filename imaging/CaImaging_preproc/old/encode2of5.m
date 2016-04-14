function code = encode2of5(n)
% This will encode a modified 2 of 5 barcode. The only modification is in
% the second guard value, which is shortened by using two narrow bars
% instead of a wide and a narrow. Narrow is 1, wide is 2. code has two
% rows: the first for bars, the second for spaces. bars is the same length
% as spaces, because spaces includes a trailing short dummy to make life
% easy.
%
% This function is based on processing the base-10 string of the number,
% which is stupid and slow and inefficient. But so is Matlab.

%% Parameters, error checking

if mod(n, 1) ~= 0
  error('encode2of5 requires integers');
end

encMat = [
  1 1 2 2 1;
  2 1 1 1 2;
  1 2 1 1 2;
  2 2 1 1 1;
  1 1 2 1 2;
  2 1 2 1 1;
  1 2 2 1 1;
  1 1 1 2 2;
  2 1 1 2 1;
  1 2 1 2 1];

guard = [1 1; 1 1];


%% Convert to string to get base-10 representation

numstr = num2str(n, '%d');

nDigits = length(numstr);

% If odd number of digits, add a leading 0
if mod(nDigits, 2) == 1
  numstr = ['0' numstr];
  nDigits = length(numstr);
end


%% Convert digit pairs

% We'll just append instead of pre-allocating, because it makes the code
% simpler and we'll only append a few times for any reasonable number.
code = [];
chunk = zeros(2, 5);
for d = 1:2:nDigits
  chunk(1, :) = encMat(str2double(numstr(d)) + 1, :);
  chunk(2, :) = encMat(str2double(numstr(d+1)) + 1, :);
  code = [code chunk];
end


%% Add guards

code = [guard code guard];


