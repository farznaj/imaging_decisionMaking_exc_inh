function tokens = simpleTokenize(str, sep)
% tokens = simpleTokenize(str, sep)
%
% This function takes a string and a separator character, and returns a
% cell array of strings divided up by the separator. If a separator is
% found at the end of str, the last entry in tokens will be an empty
% string.
%
% E.g., simpleTokenize('ab,cde,', ',') will return {'ab', 'cde', ''}.

if ~ischar(str)
  error('simpleTokenize:str must be a string');
end
if ~ischar(sep)
  error('simpleTokenize:sep must be a string');
end
if length(sep) ~= 1
  error('simpleTokenize:sep must be length one');
end

% Find indices of the separators
seps = strfind(str, sep);

% Pre-allocate token cell array
tokens = cell(1, length(seps) + 1);

if isempty(seps)
  tokens = {str};
else
  % First and last token treated directly
  tokens{1} = str(1:seps(1) - 1);
  tokens{end} = str(seps(end) + 1 : end);
  
  % Middle tokens
  if length(seps) > 1
    for t = 2:length(seps)
      tokens{t} = str(seps(t-1) + 1 : seps(t) - 1);
    end
  end
end
