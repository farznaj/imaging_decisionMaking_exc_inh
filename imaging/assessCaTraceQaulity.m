% look at traces 1 by 1 and assess trace quality

%%
p = get(0,'screensize');
figure('position',[1, p(end)-200, p(3), 120])

numUnits = size(dFOF,2);
traceQuality = NaN(1, numUnits);

for n = 1:numUnits
    set(gcf, 'name', sprintf('Unit %d/%d', n, numUnits))
    plot(dFOF(:,n))
    ylim([-.5 1.5])
    xlim([0 length(dFOF)])
    
    a = [];
    while isempty(a)
        a = input('1(good) 2(ok-good) 3(ok-bad) 4(bad)? ');
    end
    traceQuality(n) = a;
end

%%
s = [];
for u = unique(traceQuality)
    s = [s sum(traceQuality==u)];
end
s
isequal(sum(s), numUnits)