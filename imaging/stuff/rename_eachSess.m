cd \\sonas-hs.cshl.edu\churchland\data\fni17\imaging\151028

%%
a = dir('*_MCM.TIF');
aa = {a.name};
showcell(aa')

%%

for i=1:length(aa)
    o = aa{i};
    n = [aa{i}(1:end-4),'_eachSess.TIF'];
    movefile(o, n)
%     pause
end