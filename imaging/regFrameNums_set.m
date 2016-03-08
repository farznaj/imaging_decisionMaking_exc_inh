function regFrameNums = regFrameNums_set(file2read, noMotionTrs)

numFrs = frameCounts_read(file2read);

%%
numfrs_cs = cumsum(numFrs);

frs = arrayfun(@(x)numfrs_cs(x)+1 : numfrs_cs(x+1), noMotionTrs, 'uniformoutput',0);
regFrameNums = cell2mat(frs);

