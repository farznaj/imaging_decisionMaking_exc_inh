function figs_adj_poster_ax(fh,fn,axpos)
% fh: figure handle
% fn: optional, figure name for saving as pdf
% ax: optional, position of the axes
% example call: figs_adj_poster_ax(gcf)


%%
lp = get(legend, 'position');

if ~exist('axpos','var')
    axpos = [0.25    0.20    0.7000    0.7000];
end

fs = 18; % 12;
fontname = 'helvetica'; %'times';

set(fh, 'paperposition', [1 5 3.5 3.5])

% fix up axes the way I like them
set(gca,'TickDir','out','FontSize',fs,'Box','off','fontname',fontname, 'linewidth', 1);
set(gca,'position', axpos);

set(gca,'ticklength',[0.03 0.025]);


%% labels font size

xl = get(gca, 'xlabel');
set(xl, 'fontsize', fs) 
yl = get(gca, 'ylabel');
set(yl, 'fontsize', fs) 


%% tick labels
%{
a = get(gca,'ytick');
set(gca, 'ytick', a(2:2:end))

a = get(gca,'xtick');
set(gca, 'xtick', a(2:2:end))
%}

%%
set(legend, 'position', lp)
legend boxoff

%% Change line thicknesses to -1
hch = get(gca, 'children');
set(hch, 'linewidth', 1)

%% no axis
%{
% set(gca,'ytick','')
% set(gca,'xtick','')
axis 'off'
title ''
%}
%% gap between labels and axis
% set(xl,'Position',get(xl,'Position') - [0 .1 0])
% set(yl,'Position',get(yl,'Position') - [.1 0 0])

%
%{
    hAxes = gca;
%     ax = 'x';
    offsetx = 8;
    offsety = offsetx; %+35;
    
    % get label position in pixels
%     if ax=='x'

%     else

%     end
    
    % move label
%     if ax=='x'
        set(get(hAxes,'XLabel'),'Units','pixels')
        posLabel = get(get(hAxes,'XLabel'),'Position');
        set(get(hAxes,'XLabel'),'Position',posLabel+[0 -offsetx 0])
%     else
        set(get(hAxes,'YLabel'),'Units','pixels')
        posLabel = get(get(hAxes,'YLabel'),'Position');
        set(get(hAxes,'YLabel'),'Position',posLabel+[-offsety 0 0])
%     end

%     set units back to 'normalized' and 'data'
    set(hAxes,'Units','normalized')
%     if ax=='x'
        set(get(hAxes,'XLabel'),'Units','data')
%     else
        set(get(hAxes,'YLabel'),'Units','data')
%     end
    %}
    
%% save as pdf

if exist('fn', 'var')
    print('-dpdf', fn)
end
% print('-deps', fn)

