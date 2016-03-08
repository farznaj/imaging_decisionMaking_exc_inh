function [trs2incl, trs2rmv, outcomeIndx] = trialsToAnalyze(all_data, strTypes)

labels = [1 0 -1 -2 -3 -4 -5];

% suc*, fail*, early*, no dec*/cho*, wrong st*/in*, no cen* com*, no s*
% com*
labels_name = {'suc*', 'fail*', 'early*', 'no\s?[dc][eh][co]', 'wrong\s?[si][tn]', 'no\s?[c][e][n]\w*\s?[c][o][m]\w*', 'no\s?[s]\w*\s?[c][o][m]\w*'};

%{
% the following is simpler but you would need to write the outcome names precisely
% like how they appear below.
labels_name = {'success', 'failure', 'early decision', 'no decision', 'wrong start', 'no center commit', 'no side commit'};
eachstr = cellfun(@(x)strcmp(labels_name,x), strTypes, 'uniformoutput', 0)
labEachStri = cellfun(@(x)labels(x), eachstr)
%}


if strcmp(strTypes{1}, '-except')
    exc = 1;
    strTypes = strTypes(2:end);
else
    exc = 0;
end


if isnumeric(strTypes{1})
    outcomeIndx = cell2mat(strTypes);
else
    
    allStrs = cellfun(@(x)regexp(x, labels_name), strTypes, 'uniformoutput', 0);


    outcomeIndx = NaN(size(strTypes));
    for istr = 1:length(allStrs)
        outcomeIndx(istr) = labels(cellfun(@(x)~isempty(x),allStrs{istr}));
    end
end


%%
trs2ana = 1:length(all_data);
if exc % include trials that are NOT part of outcomes.
    trs2incl = trs2ana(~ismember([all_data.outcome], outcomeIndx));
    trs2rmv = trs2ana(ismember([all_data.outcome], outcomeIndx));
else % include trials that ARE part of outcomes.
    trs2incl = trs2ana(ismember([all_data.outcome], outcomeIndx));
    trs2rmv = trs2ana(~ismember([all_data.outcome], outcomeIndx));
end


