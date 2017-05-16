%% Set PMF plots for each individual mice (pooling trials of all sessions). 
% Also fit GLM.

rmvMice_5_8 = 0; % remove 5th and 8th mice below (fni16,fni19)

miceNames = {'fn03', 'fn04', 'fn05', 'fn06', 'fni16', 'fni17', 'fni18', 'fni19', 'hni01', 'hni04'};
allowCorrectResp = 'change';
uncommittedResp = 'nothing'; % 'change'; %'remove'; % % 'remove', 'change', 'nothing';
excludeShortWaitDur = true; % waitdur_th = .032; % sec  % trials w waitdur less than this will be excluded.
excludeExtraStim = false;



%%
% stimrate_all = cell(1,length(miceNames));
% y_all = cell(1,length(miceNames));
HRchoicePerc_all = cell(1,length(miceNames));
vec_rates_all = cell(1,length(miceNames));
up_all = cell(1,length(miceNames));
lo_all = cell(1,length(miceNames));
nSamples_all = cell(1,length(miceNames));

for imouse = 1:length(miceNames)
    
    % Remember data are loaded from the latest day to the earliest day.
    
    fprintf('--------------------------\n')
    mouse = miceNames{imouse};
    [day, dayLast, days2exclude] = setMouseAnalysisDays(mouse);
    
    [alldata_fileNames, days_all] = setBehavFileNames(mouse, day, dayLast, days2exclude);
    fprintf('Total number of session: %d\n', length(alldata_fileNames))
    
    [alldata, trials_per_session] = loadBehavData(alldata_fileNames);
    fprintf('Total number of trials: %d\n', length(alldata))
    %     trials_per_mouse(imouse) = length(alldata);
    
    
    %%
    [stimrate, y, HRchoicePerc, vec_rates, up, lo, nSamples] = stimrate_choice_set...
        (alldata, trials_per_session, uncommittedResp, allowCorrectResp, ...
        excludeExtraStim, excludeShortWaitDur, mouse);
    
    
    %%
    %     stimrate_all{imouse} = stimrate;
    %     y_all{imouse} = y;
    HRchoicePerc_all{imouse} = HRchoicePerc;
    vec_rates_all{imouse} = vec_rates;
    up_all{imouse} = up;
    lo_all{imouse} = lo;
    nSamples_all{imouse} = nSamples;
    
end


%% plot percentage HR vs stim rate.

plotPMF = 1;
lineColor = [.6,.6,.6]; %'k';
figure;
hold on

if plotPMF
    for imouse = 1:length(miceNames)
        
        HRchoicePerc = HRchoicePerc_all{imouse};
        vec_rates = vec_rates_all{imouse};
        up = up_all{imouse};
        lo = lo_all{imouse};
        
        %%
        wd = mode(diff(vec_rates));
        
        ee = [(HRchoicePerc-lo)', (up-HRchoicePerc)'];
        ee = ee(~isnan(HRchoicePerc),:);
        
        xx = vec_rates(~isnan(HRchoicePerc)) + wd/2;
        yy = HRchoicePerc(~isnan(HRchoicePerc));
        
        
        %%
        %         hmodal = plot(xx,yy,'.','color', lineColor);
        hmodal = plot(xx,yy,'-','color', lineColor);
        h = errorbar(xx, yy, ee(:,1), ee(:,2), 'color', lineColor,'linestyle','none');
        
        %         disp(imouse),pause
    end
end



%% Fit a GLM to the psychometric function

sigmoid = 'invgauss'; % 'logistic';  % can be 'invgauss' or 'logistic', invgauss uses probit
rates_step = 1;

xs_all =  cell(1,length(miceNames));
yfit_all =  cell(1,length(miceNames));
bs = nan(1,length(miceNames));
b_ses = nan(1,length(miceNames));
pses = nan(1,length(miceNames));

for imouse = 1:length(miceNames)
    
    HRchoicePerc = HRchoicePerc_all{imouse};
    vec_rates = vec_rates_all{imouse};
    nSamples = nSamples_all{imouse};
    
    %%
    rates = vec_rates(~isnan(HRchoicePerc)) + wd/2;
    pmf = HRchoicePerc(~isnan(HRchoicePerc));
    ns = nSamples(~isnan(HRchoicePerc));        
    
    % Sigmoid
    if length(rates) > 2
        xs = (rates(1) : rates_step : rates(end))';
        % GLM
        switch sigmoid
            case 'invgauss' % norminv(µ) = Xb
                [b, junk, stats] = glmfit(rates', pmf', 'binomial', 'link', 'probit');
                bs(imouse) = 1/b(2);
                b_ses(imouse) = stats.se(2);
                yfit = glmval(b, xs, 'probit');
                % Compute PSE. Since with probit the GLM is norminv(mu) = Xb, and
                % b(1) is for the constant term, the PSE is
                % (norminv(0.5) - b(1)) / b(2)
                pses(imouse) = (norminv(0.5, 0, 1) - b(1)) / b(2);
                
            case 'logistic' % log(µ/(1 – µ)) = Xb
                [b, junk, stats] = glmfit(rates', [pmf'.*ns' ns'], 'binomial'); % default link: logit
                bs(imouse) = 1/b(2);
                b_ses(imouse) = stats.se(2);
                yfit = glmval(b, xs, 'logit');
                % Compute PSE. Since with logit the GLM is log(mu/(1-mu)) = Xb, and
                % b(1) is for the constant term, the equation is
                % log(0.5 / (1-0.5)) = Xb,  so...
                % log(1) = Xb
                % 0 = Xb
                % ...and the PSE is
                % -b(1) / b(2)
                pses(imouse) = -b(1) / b(2);
                
            otherwise
                error('sigmoid parameter must be ''invgauss'' or ''logistic''');
        end        
        
        %%
%         plot(xs, yfit, 'color', col(imodality,:), 'LineWidth', 1);
        
        %     bestParams = fitSigConstrained(rates, pmf, ns);
        %     sigRates = rates(1)-0.5 : 0.1 : rates(end)+0.5;
        %     plot(sigRates, logistic(sigRates, bestParams), 'color', colors(imouse,:));
       
        %%
        xs_all{imouse} = xs;
        yfit_all{imouse} = yfit;
        
    end
end



%% Set a matrix for PMF and rates of all mice 

vr = unique(round(cell2mat(vec_rates_all),2)); %max(cellfun(@length, vec_rates_all));
hrp = nan(length(vr), length(miceNames));

for imouse = 1:length(miceNames)
    for ir = 1:length(vr)
        inds = vec_rates_all{imouse}==vr(ir);
        if sum(inds)>0
            hrp(ir,imouse) = HRchoicePerc_all{imouse}(inds);
        end
    end
end

hrp0 = hrp;

%% Set to nan if <50% of mice contribute to an x value
% hrp_fit = hrp0;
th = .2; .5; 
hrp(mean(~isnan(hrp),2) <= th,:) = nan;
% hrp(mean(~isnan(hrp_fit),2) <= th,:) = nan;

%% Remove bad mice
% hrp_fit = hrp0;
if rmvMice_5_8
    mouse2rmv = [5,8]; % index of mouse to remove
    hrp(:,mouse2rmv) = [];
end


%% Plot raw PMF of all mice
figure; hold on;
plot(vr+wd/2 , hrp , 'color' , [.6,.6,.6])
plot(vr+wd/2 , nanmean(hrp,2) , 'k' , 'linewidth' , 2)
xlabel('Stimulus rate (Hz)')
ylabel('Fraction high-rate choice')
xlim([min(vr)-1, max(vr+wd/2)+1])


%% Set a matrix for fitted PMF and rates of all mice

a = cellfun(@transpose,xs_all, 'uniformoutput',0);
vr_fit = unique(round(cell2mat(a),2)); %max(cellfun(@length, vec_rates_all));
hrp_fit = nan(length(vr_fit), length(miceNames));

for imouse = 1:length(miceNames)
    for ir = 1:length(vr_fit)
        inds = xs_all{imouse}==vr_fit(ir);
        if sum(inds)>0
            hrp_fit(ir,imouse) = yfit_all{imouse}(inds);
        end
    end
end

hrp0 = hrp_fit;


%% Set to nan if <50% of mice contribute to an x value
% hrp_fit = hrp0;
th = .2; .5; 
hrp_fit(mean(~isnan(hrp_fit),2) <= th,:) = nan;
% hrp(mean(~isnan(hrp_fit),2) <= th,:) = nan;

%% Remove bad mice
% hrp_fit = hrp0;
if rmvMice_5_8
    mouse2rmv = [5,8]; % index of mouse to remove
    hrp_fit(:,mouse2rmv) = [];
end


%% Plot raw pmf and GLM-fit pmf averaged across all mice

cb = 16;
y = nanmean(hrp_fit,2); ni = isnan(y); y(ni) = [];
ye = nanstd(hrp_fit,[],2); ye(ni) = [];
x = vr_fit; x(ni) = [];

figure; hold on
% plot original mean pmf 
plot(vr+wd/2 , nanmean(hrp,2), 'k.', 'markersize', 6) % 'color', [.6,.6,.6], 'linewidth' , 2)

% Plot fitted PMF
[h1,h2] = boundedline(x, y, ye, 'k', 'alpha'); %  '-b*'
set(h2, 'facecolor', [.6,.6,.6])
set(h1,'linewidth',1)

plot([cb,cb], [0,1], 'k:')
plot([min(x), max(x)],[.5, .5],'k:')

xlabel('Stimulus rate (Hz)')
ylabel('Fraction high-rate choice')
xlim([min(x)-1, max(x)+1])



%% Plot fitted PMF of each mouse, and ave mice
figure; hold on;

plot(vr_fit , hrp_fit , 'color' , [.6,.6,.6])
plot(vr_fit , nanmean(hrp_fit,2) , 'k' , 'linewidth' , 2)
%{
y = nanmean(hrp_fit,2); ni = isnan(y); y(ni) = [];
ye = nanstd(hrp_fit,[],2); ye(ni) = [];
x = vr_fit; x(ni) = [];

[h1,h2] = boundedline(x, y, ye, 'k', 'alpha'); %  '-b*'
set(h2, 'facecolor', [.6,.6,.6])
set(h1,'linewidth',2)
%}
% errorbar(x, y, ye, 'k', 'linestyle','none')



%% Look at each mouse's pmf and its glm fit.

figure; hold on
for imouse = 1:length(miceNames)
    HRchoicePerc = HRchoicePerc_all{imouse};
    vec_rates = vec_rates_all{imouse};
    
    xx = vec_rates(~isnan(HRchoicePerc)) + wd/2;
    yy = HRchoicePerc(~isnan(HRchoicePerc));

    plot(xx,yy, 'color',[.6,.6,.6])
%     plot(xs_all{imouse}, yfit_all{imouse},'color','k')
     
%     pause
end



