% 35,58

codeModuleDurs = [0.0015 0.0035];
code = cell(1,1E3);
dur_code = NaN(1,1E3);

for i = 0:1E3
    
    c = encode2of5(i);
    
    code{i+1} = encode2of5(i);    
    
    dur_code(i+1) = sum(codeModuleDurs(c(:)));
    
end

%%
figure; plot(dur_code)

% add 00 so they all have the same dur (due to the same number of digits)

