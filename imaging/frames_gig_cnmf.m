% [frn,i] = sort([12e3,18500,30e3,34e3,36e3,40e3,76e3,72e3,60e3,50e3,58e3,53e3,24e3,25e3,27e3,30e3,43e3,45e3,90e3]);
% gig = [100,126,170,200,212,235,450,420,280,308,340,317,144,160,170,178,260,275,530];
frn = [12000
       18500
       24000
       25000
       27000
       30000
       30000
       34000
       36000
       40000
       43000
       45000
       50000
       53000
       58000
       60000
       72000
       76000
       90000];
   
gig = [100
       126
       144
       160
       170
       170
       178
       200
       212
       235
       260
       275
       308
       317
       340
       280
       420
       450
       530];
   
frn(end-3)= [];
gig(end-3) = [];
frn = unique(frn);
gig = unique(gig);

x = linspace(frn(1), frn(end), 100);
y = interp1(frn, gig, x);


%%   

figure('units','normalized','outerposition',[0 0 1 1]); 
plot(x, y, '-o', 'linewidth', 2)
% plot(frn, gig, '-o', 'linewidth', 2) % plot(frn/1e3, gig, '-o'), 

grid minor
xlabel('frames'), ylabel('Giga bites')

NumTicks = 18; L = get(gca,'XLim');
set(gca,'XTick',linspace(L(1),L(2),NumTicks))

NumTicks = 20; L = get(gca,'YLim');
set(gca,'YTick',linspace(L(1),L(2),NumTicks))

n = get(gca, 'XTick');
c = num2cellOfStr(n);
xticklabels(c)

n = get(gca, 'YTick');
c = num2cellOfStr(n);
yticklabels(c)


fn = fullfile('~/Dropbox', 'frames_memory_cnmf');
savefig(fn)
print('-djpeg', fn)
