css = [0 cumsum(B_len)];
B_all(:, sum(isnan(B_all),1)==length(miceNames)) = [];


%%
B_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    B_term{ib} = B_all(:, css(ib)+1:css(ib+1));
end


%%
sea = [stats_all.se]';
% sea0 = sea;
a = find(arrayfun(@(x)isempty(x.se), stats_all));
if ~isempty(a)
    for i = 1:length(a)
        sea = insertElement(sea, a(i), nan);
    end
end

se_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    se_term{ib} = sea(:, css(ib)+1:css(ib+1));
end


%%
pa = [stats_all.p]';
pa0 = pa;
a = find(arrayfun(@(x)isempty(x.p), stats_all));
if ~isempty(a)
    for i = 1:length(a)
        pa = insertElement(pa, a(i), nan);
    end
end

p_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    p_term{ib} = pa(:, css(ib)+1:css(ib+1));
end

% if you want to know how some of the fields of stats are computed.
% isequal(sum(~isnan(y)) - size(X,2), stats.dfe)
% isequal(sqrt(diag(stats.covb)), stats.se)
% isequal(B ./ stats.se, stats.t)
%{
if estdisp
    stats.p = 2 * tcdf(-abs(stats.t), dfe);
else
    stats.p = 2 * normcdf(-abs(stats.t));
end
%}


