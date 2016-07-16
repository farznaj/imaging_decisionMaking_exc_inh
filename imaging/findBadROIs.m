% Find Eftychios's ROIs that need to be excluded.

% tau seems to be the best measure.
% tau is very related to order_measure, and using f==1 measure is enough
% (you don't need f3==1... it is almost redundant).

% corr: if u want to exclude more comps, use both corr and tau
% measures and a thresh of .4 for corr. You can go down to .3 to exclude fewer comps.
% or just go with tau (u'll exclude fewer neurons).

% remember u can set an upper limit for tau too (too slow time constants).


%%
% [A_or,C_or,S_or,P_or,srt, order_measure] = order_ROIs(A,C,S,P);
% [srt_sn,n_events,n_act] = order_components(YrA,C,[]);


%% You need to load these vars

%{
load('151102_001-002.mat', 'imHeight', 'imWidth', 'sdImage', 'Nnan_nanBeg_nanEnd')
load('151102_001-002_ch2-PnevPanResults-160624-113108.mat', 'A_m', 'C_m', 'P_m', 'badComps', 'activity_man_eftMask_ch2', 'merging_vars_m')

C = C_m;
A = A_m;
P = P_m;

nanBeg =  Nnan_nanBeg_nanEnd(2,:);
nanEnd = Nnan_nanBeg_nanEnd(3,:);
inds2rmv = cell2mat(arrayfun(@(x,y)(x:y), nanBeg, nanEnd, 'uniformoutput', 0)); % index of nan-ITIs (inferred ITIs) on C and S traces.
C(:, inds2rmv) = [];

%}


%% merge components in activity_man using mergedROIs (in case merging was done after activity was set)

if size(activity_man_eftMask_ch2,2) ~= size(C,1)
    a = activity_man_eftMask_ch2;
    a = a(:, ~badComps);
    size(a)

    clear am
    for i=1:length(merging_vars_m.merged_ROIs{1})
        am(:,i) = mean(a(:, merging_vars_m.merged_ROIs{1}{i}),2);
    end
    size(am)

    %%%%
    m = cell2mat(merging_vars_m.merged_ROIs{1});
    a(:,m) = [];
    size(a)

    %%%%
    a = cat(2, a, am);
    size(a)

    %%%%
    activity_man_eftMask_ch2 = a;
end


%% Set correlation between C and raw trace

c = corr(C', activity_man_eftMask_ch2); % this is generally higher than corr(C_df', dFOF_man)
cc = diag(c);
size(cc)
% c = corr(C_df', dFOF_man); 
% cc2 = diag(c);

figure; histogram(cc)
xlabel('Raw vs C corr')


%% Set time constants (in ms) from P.gn

frameLength = 1000/30.9; % sec.
tau = nan(size(P.gn,1), 2);
for i = 1:length(tau)
    g = P.gn{i};
    tau(i,:) = tau_d2c(g,frameLength); % tau(:,1) is rise, and tau(:,2) is decay time constant (in ms).
end

figure; histogram(tau(:,2))
xlabel('Tau\_decay (ms)')


%% Identify bad components

f1 = (tau(:,2) < 100); % 50 % (frameLength*tau(:,2) < 50) | (frameLength*tau(:,2) > 2000)
f2 = cc < .35; % .3; .4; % max([cc,cc2],[],2) < .4; % 
% f3 = full(order_measure < 2000); % very few comps with f1==0 and f3==1... ie if you go with f1==1, then f3 is more of a redundant measure.

% [sum(f1==1) sum(f2==1) sum(f3==1)]
[sum(f1==1) sum(f2==1)]
sum(f1==0 & f2==1)
sum(f1==1 | f2==1)


%% Define neurons to be excluded

badComps = (f1==1 | f2==1); % logical array same size as number of comps in C, with 1s for bad components.
num_fract_badComps = [sum(badComps) mean(badComps)]


%% Append badComps to pnevFile 

% save(pnevFileName, '-append', 'badComps')


%% Look at tau of good components

fb = find(~badComps);
[n,v] = histcounts(tau(fb,2)); figure; bar(v(1:end-1),n); hold on; plot(median(tau(fb,2)), max(n), 'r*'),  

figure; plot(tau(~badComps,2))


%%
COMs = fastCOMsA(A, [imHeight, imWidth]);
CC = ROIContoursPnevCC(A, imHeight, imWidth, .95);

% save princeton_traceQual_fni17_151102_1_2 C activity_man_eftMask_ch2 fb fg tau cc sdImage COMs CC


%% Assess the results (look at bad and good components (manual and C traces as well as contours)

fb = find(badComps); % [~,is]= sort(tau(fb,2), 'descend'); fb = fb(is); fb = fb(1:10); % fb = fb(randperm(length(fb))); 
fg = find(~badComps); % fg = fg(tau(fg,2)<600); fg = fg(2:11); % fb = fb(randperm(length(fb)));
% length(fb)


figure; 
subplot(3,2,[5]); 
imagesc(log(sdImage{2}))

for i = fb'; % fb'; %220; %477; 
    disp(i)
    hold on
    subplot(3,2,[1,2]), h1 = plot(C(i,:)); 
    title(sprintf('tau = %.2f ms', tau(i,2))),  % title(sprintf('%.2f, %.2f', [cc(i), tau(i,2)])), 
    xlim([1 size(C,2)])
    ylabel('C')% (denoised-demixed trace)')
    
    subplot(3,2,[3,4]), h2 = plot(activity_man_eftMask_ch2(:,i)); 
    title(sprintf('corr = %.2f', cc(i))),  
    xlim([1 size(C,2)])
    ylabel('Raw') % (averaged pixel intensities)')
    
    subplot(3,2,[5]); hold on
    h3 = plot(CC{i}(2,:), CC{i}(1,:), 'r');
    xlim([COMs(i,2)-50  COMs(i,2)+50])
    ylim([COMs(i,1)-50  COMs(i,1)+50])
%     imagesc(reshape(A(:,i), imHeight, imWidth))
    
    pause
    delete([h1,h2,h3])
end



%% Plot COMs of bad and good components on the medImage.

im = sdImage{2}; % medImage{2};

COMs = fastCOMsA(A, size(im));

% bad components
figure; 
imagesc(im)
hold on
for rr = find(badComps')
    plot(COMs(rr,2), COMs(rr,1), 'r.')
end

% good components
figure; 
imagesc(im)
hold on
for rr = find(~badComps')
    plot(COMs(rr,2), COMs(rr,1), 'r.')
end


%%
A = A(:, ~badComps);
C = C(~badComps,:);
C_df = C_df(~badComps,:);


%%
%{
frmv = sum([f1,f2,f3], 2); % or just go with tau and correlation.

s = [];
for i=0:3
    s = [s, sum(frmv==i)];
end
s

excl = find(frmv==2); % bad components
excl = find(frmv==0); % good components.


excl = find(f1==1); % & frmv~=2)
%}

