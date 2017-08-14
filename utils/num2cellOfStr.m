function c = num2cellOfStr(n)
% convert numerical array "n" into a same-size cell array of strings "c"
% e.g.
%{
n =  1.0e+04 * [1.0000    1.4706    1.9412    2.4118];
c = num2cellOfStr(n)
c = {'10000 '    '14706 '    '19412 '    '24118 '};
%}

% xx = get(gca, 'XTick');
c = cellfun(@(x)sprintf('%.0f ',x), num2cell(n,1), 'uniformoutput', 0);
% xticklabels(aa)
