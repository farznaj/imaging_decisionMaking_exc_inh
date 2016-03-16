function numFrs = frameCountsRead(frameCountFileName)
% numFrs = frameCountsRead(frameCountFileName); 
% This function reads the frameCounts text file (frameCountFileName), and
% outputs number of frames (numFrs) for each trial.
%
% Based on matt's code.

fid = fopen(frameCountFileName, 'r');

numFrs = [];
line = 0;
while 1
    line = line+1;
    oneline = fgetl(fid);
    if oneline==-1
        break
    end
    tokens = simpleTokenize(oneline, ' ');
    numFrs = [numFrs str2double(tokens{1})];
end

