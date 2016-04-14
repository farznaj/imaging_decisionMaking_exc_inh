function axprefs(ax,fs)
if nargin == 1
  fs = 12;
end;

% fix up axes the way I like them
set(ax,'TickDir','out','FontSize',fs,'Box','off','fontname','times');


set(gca,'ticklength',[0.03 0.025]);
