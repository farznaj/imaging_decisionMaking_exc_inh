function regFrameNums = frameNumsSet(frameCountFileName, trNums)
% regFrameNums = regFrameNumsSet(file2read, noMotionTrs)
%
% Takes the frameCount text file (frameCountFileName) and trial numbers
% (trNums), and returns the frames corresponding to those trials.

numFrs = frameCountsRead(frameCountFileName);

%%
numfrs_cs = [0 cumsum(numFrs)];

frs = arrayfun(@(x)numfrs_cs(x)+1 : numfrs_cs(x+1), trNums, 'uniformoutput',0);
regFrameNums = cell2mat(frs);

