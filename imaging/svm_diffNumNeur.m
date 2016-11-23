a = cellfun(@(x)mean(x,2), perClassErrorTest_data_nN_all, 'uniformoutput', 0);

figure; boundedline(1:length(b), cellfun(@mean, a), cellfun(@std, a), 'alpha')

a = cellfun(@(x)mean(x,2), perClassErrorTest_shfl_nN_all, 'uniformoutput', 0);
hold on; boundedline(1:length(b), cellfun(@mean, a), cellfun(@std, a), 'r', 'alpha')
