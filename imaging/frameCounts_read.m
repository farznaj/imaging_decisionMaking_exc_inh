function numFrs = frameCounts_read(file2read)

fid = fopen(file2read, 'r');

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

