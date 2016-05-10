function [xnew, ynew] = plotcone(y, x)
% Useful when all elements in y have the same x, and you want to plot y vs
% x. This function gives x_new and y_new that make the y vs x plot look
% like a nice gaussian cone (by adding small random values to the x of each
% element to separate the points).

%%
y = reshape(y, [],1);

v = linspace(min(y), max(y)+.001, 10);
[~,ib] = histc(y, v);
wd = linspace(.01, .2, 10/2)/2;
% wd = linspace(.01, .2, 10/2);
wd = [wd, wd(end:-1:1)];

% figure; hold on
xnew = []; ynew = [];
%     xnsa = []; ynsa = [];
for iv = 1:length(v)-1
    iwd = wd(iv);
    x2 = x-iwd + (2*iwd)*rand(1, sum(ib==iv));
%             y = plca(ib==iv);
%             plot(x,y,'k.')
    ys = y(ib==iv)';
    xs = x2(1:length(ys));
%         yns = plca(ib==iv & pks'>.05)';        
%         xns = x(length(ys)+1:length(ys)+length(yns));

    xnew = [xnew; xs'];
    ynew = [ynew; ys'];

%     if ~isempty(ys)
%         plot(xs , ys ,  'marker', 'o', 'markersize', 3, 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'linestyle', 'none')
%     end
end

xnew(end+1:length(y)) = NaN;
ynew(end+1:length(y)) = NaN;

