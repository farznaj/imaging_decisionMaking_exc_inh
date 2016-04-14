function [numbers, nRepaired] = repair2of5Sweep(numbers, codes)
% [numbers, nRepaired] = repair2of5Sweep(numbers, codes)
%
% Attempt to repair damaged trial codes, assuming the front end got clipped
% off by too long of a latency to start scanning.
%
% This algorithm performs a forward pass and then a backward pass. The
% forward pass finds each time a known code is followed by a bad one. It
% then finds the next known code. Based on the number of bad codes between
% this bad code and the next known code, it can narrow down the range of
% possible codes this one could be. It then compares this code with all the
% possible candidates. If there's exactly one match, that's the answer. The
% backwards pass does the same thing, but looks at codes that immediately
% precede a known code.
%
% Added bonus: if there's a damaged code flanked by two known codes that
% differ by exactly 2, we can infer that this code should have been the
% value between them. This will throw a warning, though, to let you know
% something weird happened.
%
% Iteration until nRepaired is 0 is recommended.


nRepaired = 0;


%% Forward pass

good = (numbers > 0);

% Cannot check anything until first good value
tr = find(good, 1);
while tr < length(numbers) - 1
  tr = tr + 1;
  
  % If this number is good, move on
  if good(tr)
    continue;
  end
  
  % Find bounds
  % If we're here, the previous number was good
  num1 = numbers(tr-1);
  tr2 = tr + find(numbers(tr+1:end) > 0, 1);
  
  % If we didn't find another good number after this, we're done
  if isempty(tr2)
    break;
  end
  
  num2 = numbers(tr2);
  
  % Grab code we're working on
  codeToID = codes{tr};
  
  % Clip off front pair for code, unless there was a long high, since
  % otherwise we can't know if we clipped it and got it wrong
  if codeToID(1) ~= 2
    if size(codeToID, 2) > 2
      codeToID = codeToID(:, 2:end);
    else
      % All codes end this way
      codeToID = [1; 1];
    end
  
  end
  
  % Figure out what codes the damaged one might be
  nCodesInInt = tr2 - tr;
  candidates = num1 + 1 : num2 - nCodesInInt;
  
  candCodes = arrayfun(@encode2of5, candidates, 'UniformOutput', false);
  
  % If any candidates produce codes that are too short, those are wrong
  longEnough = cellfun(@(c) size(c, 2), candCodes) >= size(codeToID, 2);
  candCodes = candCodes(:, longEnough);
  candidates = candidates(longEnough);
  
  % Trim candidates to the right size
  candCodes = cellfun(@(c) c(:, end-size(codeToID, 2)+1:end), candCodes, 'UniformOutput', false);
  
  matches = find(cellfun(@(c) isequal(c, codeToID), candCodes));
  
  if length(matches) == 1
    % Success!
    numbers(tr) = candidates(matches);
    nRepaired = nRepaired + 1;
  elseif length(candidates) == 1 && isempty(matches)
    % Can infer, but something is wrong with code
    numbers(tr) = candidates;
    nRepaired = nRepaired + 1;
    warning('Something went wrong internal to a code, but it could be inferred. Still, worry a little.');
  else
    % Failure, skip to next good value, then back up one so we advance to
    % the correct one next iteration
    tr = tr + find(good(tr+1:end), 1) - 1;
  end
end


good = (numbers > 0);


%% Backward pass

% Cannot check anything until last good value
tr = find(good, 1, 'last');
while tr > 2
  tr = tr - 1;
  
  % If this number is good, move on
  if good(tr)
    continue;
  end
  
  % Find bounds
  % If we're here, the next number was good
  num2 = numbers(tr+1);
  
  tr1 = find(numbers(1:tr-1) > 0, 1, 'last');
  
  % If we didn't find another good number after this, we're done
  if isempty(tr1)
    break;
  end
  
  num1 = numbers(tr1);
  
  % Grab code we're working on
  codeToID = codes{tr};
  
  % Clip off front pair for code, unless there was a long high, since
  % otherwise we can't know if we clipped it and got it wrong
  if codeToID(1) ~= 2
    if size(codeToID, 2) > 2
      codeToID = codeToID(:, 2:end);
    else
      % All codes end this way
      codeToID = [1; 1];
    end
  
  end
  
  % Figure out what codes the damaged one might be
  nCodesInInt = tr - tr1;
  candidates = num1 + nCodesInInt : num2 - 1;
  
  candCodes = arrayfun(@encode2of5, candidates, 'UniformOutput', false);
  
  % If any candidates produce codes that are too short, those are wrong
  longEnough = cellfun(@(c) size(c, 2), candCodes) >= size(codeToID, 2);
  candCodes = candCodes(:, longEnough);
  candidates = candidates(longEnough);
  
  % Trim candidates to the right size
  candCodes = cellfun(@(c) c(:, end-size(codeToID, 2)+1:end), candCodes, 'UniformOutput', false);
  
  matches = find(cellfun(@(c) isequal(c, codeToID), candCodes));
  
  if length(matches) == 1
    % Success!
    numbers(tr) = candidates(matches);
    nRepaired = nRepaired + 1;
  else
    % Failure, skip to next good value, then back up one so we advance to
    % the correct one next iteration
    tr = find(good(1:tr-1), 1, 'last') + 1;
  end
end