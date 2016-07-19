% Get rid of bleedthrough from greem to red channel by doing a regression
% between red and green channel activities (Matt's suggestion). This will
% give us an offset, ie the true red signal that we are interseted in, and
% a slope, ie how much of the green signal gets to the red channel (this is
% due to the properties of the red-channel filters and dichroic mirros that
% pick up part of the green signal.)


% is noise really the same for both channels? ... doesn't seem like. It
% seems to be less for the green channel.

% since the brightness of the 2 channels is different, should I normalize
% to max the activity of each neuron? ... doesn't seem like you need to do
% this.

% does regress understimate or overstimate it? ... yes seems it is actually understimating.

% how about u check the other method that matt suggested too.

% is this amount of variaton in slope ok? brighter neurons : higher slope, lower offset.

% read about bleedthrough... does it depend on the intensity? the slope
% doesn't depend on signal intensity... it is determined by the filter and
% dichroic mirror properties.


%% check the slope equation for pca

x = (1:1000) + 100*randn(1,1000);
% y = (1:1000)*4 + 100*randn(1,1000) + 500;
y = (500 + .5 * x) + 100*randn(1,1000);
figure; plot(x, y, '.')


% pca
pcs = pca([x' y']);
slp = pcs(2,1) / pcs(1,1);

% regression
glmfit(x,y)
% md1 = fitglm(x,y);

figure; hold on
plot(x, y, '.')
xy = [x(:); y(:)];
plot([min(xy)  max(xy)] , [min(xy)  max(xy)] , 'b')
title([slp, md1.Coefficients.Estimate(2)])


%%
%%%%%%%%%%%%%%% Analysis starts here %%%%%%%%%%%%%%%

load('151101_001.mat', 'pmtOffFrames', 'imHeight', 'imWidth', 'medImage'); 
load('151101_001_ch2-PnevPanResults-160422-081351.mat', 'C_df'); 
load('151101_001_ch2-PnevPanResults-160422-081351.mat', 'activity_man_eftMask_ch1', 'activity_man_eftMask_ch2'); 
load('151101_001_ch2-PnevPanResults-160422-081351.mat', 'A', 'C', 'b', 'f')


%%
smoothPts = 6; minPts = 7000; %800;

df_man_eftMask_ch1 = konnerthDeltaFOverF(activity_man_eftMask_ch1, pmtOffFrames{1}, smoothPts, minPts);
df_man_eftMask_ch2 = konnerthDeltaFOverF(activity_man_eftMask_ch2, pmtOffFrames{2}, smoothPts, minPts);


%%
trace_ch1 = activity_man_eftMask_ch1;
trace_ch2 = activity_man_eftMask_ch2;

trace_ch1 = df_man_eftMask_ch1;
trace_ch2 = df_man_eftMask_ch2;

trace_ch1 = df_man_eftMask_ch1;
trace_ch2 = C_df(1:end-1,:)';


%% Compute correlation between ch1 and ch2 activity for each ROI.

crr = NaN(1, nn);
for rr = 1:nn
    t1 = trace_ch1(:,rr);
    t2 = trace_ch2(:,rr);
    crr(rr) = corr(t1, t2);
end

figure; plot(crr)


%%
nn = size(trace_ch1, 2);

mu_ch1 = mean(trace_ch1, 1);
mu_ch2 = mean(trace_ch2, 1);


figure;
subplot(321), hold on
plot(mean(trace_ch1, 2))
plot(mean(trace_ch2, 2))
xlabel('frame')
ylabel('mean activity of all neurons')
legend('ch1', 'ch2')

subplot(322), hold on
m1 = mean(trace_ch1(:,1:20), 2);
m2 = mean(trace_ch2(:,1:20), 2);
plot(m1)
plot(m2)
legend('ch1', 'ch2')
title('mean activity of 1st 20 neurons')

subplot(323), hold on
plot(mean(trace_ch1(:,end-19:end), 2))
plot(mean(trace_ch2(:,end-19:end), 2))
legend('ch1', 'ch2')
title('mean activity of last 20 neurons')


subplot(324), hold on
errorbar(mu_ch1, std(trace_ch1, [], 1))
xlabel('neuron')
ylabel('mean +/- std of activity')
title('ch1')

subplot(325), hold on
errorbar(mu_ch2, std(trace_ch2, [], 1))
xlabel('neuron')
ylabel('mean +/- std of activity')
title('ch2')


subplot(326), hold on
rr = 1:2;
x = trace_ch2(:, rr); 
y = trace_ch1(:, rr);
plot(x, y, 'k.')

rr = nn-1:nn;
x = trace_ch2(:, rr); 
y = trace_ch1(:, rr);
plot(x, y, 'g.')

xlabel('ch2')
ylabel('ch1')
legend('k: 1st 20', 'g: last 20')
title('activity')


%% Normalize the signal ?

normt1 = mean(trace_ch1);
trace_ch1 = bsxfun(@rdivide, trace_ch1, normt1);

% for rr = 1 : nn
%     trace_ch1(:,rr) = shiftScaleY(trace_ch1(:,rr));
% end


normt2 = mean(trace_ch2);
trace_ch2 = bsxfun(@rdivide, trace_ch2, normt2);

% for rr = 1 : nn
%     trace_ch2(:,rr) = shiftScaleY(trace_ch2(:,rr));
% end


%% model: red = offset + slope * green

% Use PCA (aassumption: noise on x and y data is equal).

slope = NaN(1, nn);
for rr = 1 : nn
%     x_y = [trace_ch1(:,rr) trace_ch2(:,rr)];
    x_y = [trace_ch2(:,rr) trace_ch1(:,rr)];
    pcs = pca(x_y);
    slope(rr) = pcs(2,1) / pcs(1,1);    
end

% Compute the offset term
% offset = mu_ch2 - slope.*mu_ch1;
offset = mu_ch1 - slope.*mu_ch2;




% Use regression (assumption: x data doesn't have much noise).

slope = NaN(1, nn);
offset = NaN(1, nn);
for rr = 1 : nn
    %
    % model: red = offset + slope * green
    % y: red, x: green
    x = trace_ch2(:,rr); 
    y = trace_ch1(:,rr);
%     pp(:,rr) = glmfit(x,y);  % fitglm(x,y);
    p = regress(y, [x, ones(size(x,1),1)]);
    slope(rr) = p(1);
    offset(rr) = p(2);
    %}

    %{
    % model: green = offset2 + slope2 * red
    % y: green, x: red
    x = trace_ch1(:,rr); 
    y = trace_ch2(:,rr);
    p = regress(y, [x, ones(size(x,1),1)]);    
    slope(rr) = 1 / p(1);
    offset(rr) = -slope(rr) * p(2);
    %}
end

%{
p = regress(red, [green, ones(size(green, 1), 1)]);
slope = p(1);
intercept = p(2);

p = regress(green, [red, ones(size(red, 1), 1)]);
slope = 1 / p(1);
intercept = -slope * p(2);
%}






%% Plot slope and offset

% figure('name', 'slope'); 
figure;
subplot(421), plot(slope), title('slope')
subplot(423), histogram(slope)
subplot(425), errorbar(mean(slope), std(slope))
regress(slope', [(1:length(slope))', ones(length(slope), 1)])

% figure('name', 'offset'); 
subplot(422), plot(offset), title('offset')
subplot(424), histogram(offset)
subplot(426), errorbar(mean(offset), std(offset))
subplot(428), hold on, plot(mu_ch1), plot(mu_ch2), title('mu')
regress(offset', [(1:length(offset))', ones(length(offset), 1)])


%% Use Efty's A, C to set the median image of channel 2.

medc = median(C, 2);
% medc = slope' .* medc;

medEfty = A*medc + b*median(f);
medEfty = reshape(medEfty, imHeight, imWidth);

% mdd = medImage{1} - medEfty;
% figure; imagesc(mdd), 
% caxis([0 max(mdd(:))])


%%
figure; 
subplot(221), imagesc(medImage{1}), freezeColors, colorbar, title('ch1')
subplot(222), imagesc(medImage{2}), freezeColors, colorbar, title('ch2')
subplot(224), imagesc(medEfty), freezeColors, colorbar, title('ch2 Efty')


%% Compute the clean median image for ch1, ie the image that does not have the bleedthrough signal anymore!
% model: red = offset + slope * green
% so: med(red) = offset + slope * med(green)
% so offset = med(red) - slope * med(green)

figure; 
ha = tight_subplot(2,2,[.05],[.05],[.05]);

md = medImage{1} - medImage{2};
axes(ha(1)); imagesc(md), freezeColors, colorbar

mdd = medImage{1} - medEfty;
axes(ha(3)); imagesc(mdd), freezeColors, colorbar

% md2 is what I will use to identify inhibit neurons... this is supposedly
% the image that is free of the effect of bleedthrough.
md2 = medImage{1} - mean(slope)*medImage{2}; 
axes(ha(2)); imagesc(md2), freezeColors, colorbar
% caxis([0 max(md2(:))])


md3 = medImage{1} - mean(slope)*medEfty;
axes(ha(4)); imagesc(md3), freezeColors, colorbar
% caxis([0 max(md3(:))])

% figure; imagesc(md3)



