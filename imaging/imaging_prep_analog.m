filename = 'Z:\calcium-imaging\Farzaneh\FNI11\150606\150606_001.bin';

%%
[vals, sampFreq] = readMOMAnalog(filename, 1);

figure; plot(vals');

%%
volt = vals(1,:); % analog data of 1st channel : trial code
figure; plot(volt);

[numbers, codeTimes] = segmentVoltageAndReadBarcodes(volt);

%%


