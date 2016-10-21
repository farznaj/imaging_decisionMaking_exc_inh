function [] = saveFig2Directory(Dir, hf)
if ~exist('hf','var')
   hf = []; 
end
n = 1;
Directory = sprintf([Dir '_%03d/'], n);  % Finds the subdirectory in which the files will be placed.
while exist(Directory,'dir')
    n = n+1;
    Directory = sprintf([Dir '_%03d/'], n);  % Finds the subdirectory in which the files will be placed.
end
mkdir(Directory)
if isempty(hf)
    hf = gcf;
    for i=1:hf.Number
        h = figure(i);
    %     set(h(i),'InvertHardcopy','off','PaperPositionMode','auto','color', 'none')
        saveas(h,[Directory num2str(i)],'fig')
        saveas(h,[Directory num2str(i)],'epsc')
        saveas(h,[Directory num2str(i)],'svg')
        saveas(h,[Directory num2str(i)],'pdf')
    end
    
 
else
for i=1:length(hf)
%     set(hf(i)(i),'InvertHardcopy','off','PaperPositionMode','auto','color', 'none')
        saveas(hf(i),[Directory num2str(i)],'fig')
        saveas(hf(i),[Directory num2str(i)],'epsc')
        saveas(hf(i),[Directory num2str(i)],'svg')
        saveas(hf(i),[Directory num2str(i)],'pdf')
end
end
end