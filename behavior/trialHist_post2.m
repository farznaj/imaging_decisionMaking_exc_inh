css = [0 cumsum(B_len)];

B_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    B_term{ib} = B_all(:, css(ib)+1:css(ib+1));
end

sea = [stats_all.se]';
se_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    se_term{ib} = sea(:, css(ib)+1:css(ib+1));
end

pa = [stats_all.p]';
p_term = cell(1, length(B_len));
for ib = 1:length(B_len)
    p_term{ib} = pa(:, css(ib)+1:css(ib+1));
end