function plot_mtr(X, trial_times, border)

if ~exist('border','var')
    border = 1;
end

[N,T] = size(X);  % number of traces to plot

% figure;

plot(X' + repmat(border*(0:N-1),T,1)); hold on;
for i = 2:length(trial_times)
    plot([trial_times(i),trial_times(i)],[0,border*N],':', 'color',[.6,.6,.6]); hold on;
end
hold off;
