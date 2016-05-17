% the required inputs are driven by either of the following :
% trialHist_prep
% trialHist_micePooled

% this is a quick thing for now, later you need to do this properly, taking
% into account p values of each mouse for each coefficient, and/or number
% of trials that get fitted in logistRegress for each coef.
q = input('specify outliers\n');

if ~isempty(q)
    error('fill in here!')
    
    % remove allowCorrect
    outlier_s = find(sum(p_term{3}>.05, 2)==2);
    B_term{3}(outlier_s,:) = NaN;
    
    outlier_f = 1; % find(sum(p_term{4}>.05, 2)==2)
    B_term{4}(outlier_f,:) = NaN;
    
    % change allowCorrect
    outlier_s = 7; % find(sum(p_term{3}>.05, 2)==2);
    B_term{3}(outlier_s,:) = NaN;
    
    outlier_f = [2, 4]; % find(sum(p_term{4}>.05, 2)==2)
    B_term{4}(outlier_f,:) = NaN;
    
    % nothing allowCorrect
    outlier_s = [4, 7]; % find(sum(p_term{3}>.05, 2)==2);
    B_term{3}(outlier_s,:) = NaN;
    
    outlier_f = 4; % find(sum(p_term{4}>.05, 2)==2)
    B_term{4}(outlier_f,:) = NaN;
    
else
    
    %%
    if all([binITIs, binRates])
        
        n_bTerm = cellfun(@(x)sum(~isnan(x)), B_term, 'uniformoutput', 0);
        for ib = 1:length(n_bTerm)
            fprintf(['num valid elements in each bin of term %d = ', repmat('%d ',1,length(n_bTerm{ib})), '\n'], ib, n_bTerm{ib})
        end

        disp('h_p for ttest2 (across mice) between the 1st and last column of ')
        for ib = 1:length(B_term)            
            [h, p] = ttest2(B_term{ib}(:,1), B_term{ib}(:,end));
            h_P = [h, p];
            fprintf('term %i:\n', ib)
            disp(h_P)
        end
        
        %{
        numR = sum(~isnan(B_rate_all));
        numIS = sum(~isnan(B_itiS_all));
        numIF = sum(~isnan(B_itiF_all));

        fprintf(['numTrs in each rate bin: ', repmat('%d ',1,length(numR)), '\n'], numR)
        fprintf(['numTrs in each ITI_succ bin: ', repmat('%d ',1,length(numIS)), '\n'], numIS)
        fprintf(['numTrs in each ITI_fail bin: ', repmat('%d ',1,length(numIF)), '\n'], numIF)
        %}
        
        
        %% plot rate and iti coeffs.
        
        col = {'k','r','g', 'b', 'm', 'c'};
        figure;
        for ib = 1:length(B_term)
            subplot(length(B_term),1,ib)
            h = errorbar(1:size(B_term{ib},2), nanmean(B_term{ib},1), nanstd(B_term{ib},[],1)./sqrt(n_bTerm{ib}), '.', 'color', 'k', 'markersize', 10); %, col{ib});
%             errorbar_tick(h, 15)
            ylabel('regress coeff')
        end
        
        %{
    figure;
    subplot 211
    errorbar(vec_ratesdiff_all, nanmean(B_rate_all,1), nanstd(B_rate_all,[],1)./sqrt(numR), '.k');
    xlabel('abs(stim rate - cb)')
    ylabel('regress coeff')
    
    subplot 212, hold on
    errorbar(vec_iti_all(1:size(B_itiS_all,2)), nanmean(B_itiS_all,1), nanstd(B_itiS_all,[],1)./sqrt(numIS), '.k')
    errorbar(vec_iti_all(1:size(B_itiF_all,2)), nanmean(B_itiF_all,1), nanstd(B_itiF_all,[],1)./sqrt(numIF), '.r')
    set(gca, 'xtick', vec_iti_all(1:size(B_itiF_all,2)))
    set(gca, 'xticklabel', {['<', num2str(vec_iti(2))], ['>=', num2str(vec_iti(2))]})
    xlabel('ITI (sec)')
    ylabel('regress coeff')
        %}
        
        %% plot a histogram of B's
        %{
    figure
    % subplot 211
    % ITI_Succ
    v =linspace(min(B_itiS_all(:)), max(B_itiS_all(:)), 10);
    n = histc(B_itiS_all, v); % [n1, edges, bin] = histcounts(B_itiS_all(:,1), v, 'normalization', 'probability');
    n = bsxfun(@rdivide, n, numIS);
    subplot(211), hold on
    plot(v,n(:,1),'k-', v,n(:,2),'k-.')
    subplot(212), hold on
    plot(v, cumsum(n(:,1)),'k-', v, cumsum(n(:,2)),'k-.')
    
    % subplot 212
    % ITI_Failure
    v =linspace(min(B_itiF_all(:)), max(B_itiF_all(:)), 10);
    n = histc(B_itiF_all, v);
    n = bsxfun(@rdivide, n, numIF);
    subplot(211)
    plot(v,n(:,1),'r-', v,n(:,2),'r-.')
    subplot(212)
    plot(v, cumsum(n(:,1)),'r-', v, cumsum(n(:,2)),'r-.')
        %}
        
        %% look at average p values of each term
        
        for iterm = 1:length(se_term)
            a = nanmean(p_term{iterm},1);
            fprintf(['Average P of term %d =', repmat('%.2f  ', [1, length(a)]), '\n'], iterm, a)
        end
        
        
        %% plot B for previous outcome after long vs. short iti. or strong vs weak current stim.
        
        % or previous outcome after strong vs. weak stim.
        a = nanmean(B_term{3},1);
        b = nanmean(B_term{4},1);
        mn3 = min(B_term{3}(:));
        mx3 = max(B_term{3}(:));
        mn4 = min(B_term{4}(:));
        mx4 = max(B_term{4}(:));
        %       H=errorbarxy(x,y,xe,ye,{'ko-', 'b', 'r'});
        figure;
        subplot(211)
        try
            h = errorbarxy(B_term{3}(:,1), B_term{3}(:,end), se_term{3}(:,1), se_term{3}(:,end), {'k.','k','k'});
            set(findobj(gca, 'marker', '.'), 'markersize', 12)
            set(findobj(gca, 'type', 'line', 'color', 'k'), 'linewidth', 1.5)
            %         delete(findobj(gca, 'color', 'b'))
            %         set(findobj(gca, 'color', 'r'), 'color', 'k')
        catch
        end
        hold on
%         plot(B_term{3}(:,1), B_term{3}(:,end), 'k.', 'markersize', 8)
        plot(a(1), a(end), 'r*')
        plot([mn3 mx3], [mn3 mx3], '--', 'color', [.6 .6 .6])
        title('B\_iti\_s')
        if regexp(regressModel, 'outcomeITI')
            xlabel('B of short ITI')
            ylabel('B of long ITI')
        elseif regexp(regressModel, 'outcomeRate')
            ylabel('B of strong stim')
            xlabel('B of weak stim')
        end
        box off
        set(gca,'tickdir','out')
        set(gca,'ticklength',[.025 0])
        
        
        subplot(212)
        try
            h = errorbarxy(B_term{4}(:,1), B_term{4}(:,end), se_term{4}(:,1), se_term{4}(:,end), {'k.','k','k'});
            set(findobj(gca, 'marker', '.'), 'markersize', 12)
            set(findobj(gca, 'type', 'line', 'color', 'k'), 'linewidth', 1.5)
            %         errorbarxy(B_term{4}(:,1), B_term{4}(:,end), se_term{4}(:,1), se_term{4}(:,end))
            %         delete(findobj(gca, 'color', 'b'))
            %         set(findobj(gca, 'color', 'r'), 'color', 'k')
        catch
        end
        hold on
%         plot(B_term{4}(:,1), B_term{4}(:,end), 'k.', 'markersize', 8)
        plot(b(1), b(end), 'r*')
        plot([mn4 mx4], [mn4 mx4], '--', 'color', [.6 .6 .6])
        title('B\_iti\_f')        
        if regexp(regressModel, 'outcomeITI')
            xlabel('B of short ITI')
            ylabel('B of long ITI')
        elseif regexp(regressModel, 'outcomeRate')
            ylabel('B of strong stim')
            xlabel('B of weak stim')
        end
%         xlim([0 1.5])
%         ylim([-.4 1.8])
        box off
        set(gca,'tickdir','out')
        set(gca,'ticklength',[.025 0])

        
        %% summary plots of the conventinal analysis of trial history.
        
        mn = min([fract_change_choosingSameChoice_aftS_all(:); fract_change_choosingSameChoice_aftF_all(:); fract_change_choosingHR_aftHR_vs_LR_S_all(:); fract_change_choosingLR_aftLR_vs_HR_S_all(:); fract_change_choosingHR_aftHR_vs_LR_F_all(:); fract_change_choosingLR_aftLR_vs_HR_F_all(:)]);
        mx = max([fract_change_choosingSameChoice_aftS_all(:)]);
%         sum(isnan(fract_change_choosingLR_aftLR_vs_HR_F_all),2)==0 % you want to include only these indeces for computing mn and mx.
%         mx = max([fract_change_choosingSameChoice_aftS_all(:); fract_change_choosingSameChoice_aftF_all(:); fract_change_choosingHR_aftHR_vs_LR_S_all(:); fract_change_choosingLR_aftLR_vs_HR_S_all(:); fract_change_choosingHR_aftHR_vs_LR_F_all(:); fract_change_choosingLR_aftLR_vs_HR_F_all(:)]);
        lv = size(fract_change_choosingHR_aftHR_vs_LR_F,1);
        
        % previous outcome shown for strong vs weak current rate
        % if there is only a single iti bin but several rate bins go with this plot.
        if size(fract_change_choosingSameChoice_aftS_all,2)==1
            % loop over and make a figure for each rate bin in case binningRates for conventional analysis was set to 1.
            % showing long vs short iti.
            %         for irv = 1:lv
            
            xweak = 1 : lv : size(fract_change_choosingSameChoice_aftS_all,1);
            xstrong = lv : lv : size(fract_change_choosingSameChoice_aftS_all,1);
            
            figure('name', 'Relative change in choosing ...');
            
            %%%% success
            subplot(231)
            hold on
            plot(fract_change_choosingSameChoice_aftS_all(xweak,1), fract_change_choosingSameChoice_aftS_all(xstrong,end), 'k.')
            plot([mn mx], [mn mx])
            ylabel({'Succ', 'Strong stim'})
            title('Same vs different side')
            
            subplot(232)
            hold on
            plot(fract_change_choosingHR_aftHR_vs_LR_S_all(xweak,1), fract_change_choosingHR_aftHR_vs_LR_S_all(xstrong,end), 'k.')
            plot([mn mx], [mn mx])
            title('HR vs LR aft HR')
            
            subplot(233)
            hold on
            plot(fract_change_choosingLR_aftLR_vs_HR_S_all(xweak,1), fract_change_choosingLR_aftLR_vs_HR_S_all(xstrong,end), 'k.')
            plot([mn mx], [mn mx])
            title('LR vs HR aft LR')
            
            
            %%%% failure
            subplot(234)
            hold on
            plot(fract_change_choosingSameChoice_aftF_all(xweak,1), fract_change_choosingSameChoice_aftF_all(xstrong,end), 'r.')
            plot([mn mx], [mn mx])
            ylabel({'Fail', 'Strong stim'})
            title('Same vs different side')
            
            subplot(235)
            hold on
            plot(fract_change_choosingHR_aftHR_vs_LR_F_all(xweak,1), fract_change_choosingHR_aftHR_vs_LR_F_all(xstrong,end), 'r.')
            plot([mn mx], [mn mx])
            title('HR vs LR aft HR')
            xlabel('Weak stim')
            
            subplot(236)
            hold on
            plot(fract_change_choosingLR_aftLR_vs_HR_F_all(xweak,1), fract_change_choosingLR_aftLR_vs_HR_F_all(xstrong,end), 'r.')
            plot([mn mx], [mn mx])
            title('LR vs HR aft LR')
            %         end
            
        else
            
            % previous outcome shown for long vs short iti.
            % loop over and make a figure for each rate bin in case binningRates for conventional analysis was set to 1.
            % showing long vs short iti.
            for irv = 1:lv
                
                xi = irv : lv : size(fract_change_choosingSameChoice_aftS_all,1);
                
                figure('name', sprintf('Rate bin %d, relative change in choosing ...', irv));
                
                %%%% success
                subplot(231)
                hold on
                plot(fract_change_choosingSameChoice_aftS_all(xi,1), fract_change_choosingSameChoice_aftS_all(xi,end), 'k.')
                plot([mn mx], [mn mx])
                ylabel({'Succ', 'Long ITI'})
                title('Same vs different side')
                
                subplot(232)
                hold on
                plot(fract_change_choosingHR_aftHR_vs_LR_S_all(xi,1), fract_change_choosingHR_aftHR_vs_LR_S_all(xi,end), 'k.')
                plot([mn mx], [mn mx])
                title('HR vs LR aft HR')
                
                subplot(233)
                hold on
                plot(fract_change_choosingLR_aftLR_vs_HR_S_all(xi,1), fract_change_choosingLR_aftLR_vs_HR_S_all(xi,end), 'k.')
                plot([mn mx], [mn mx])
                title('LR vs HR aft LR')
                
                
                %%%% failure
                subplot(234)
                hold on
                plot(fract_change_choosingSameChoice_aftF_all(xi,1), fract_change_choosingSameChoice_aftF_all(xi,end), 'r.')
                plot([mn mx], [mn mx])
                ylabel({'Fail', 'Long ITI'})
                title('Same vs different side')
                
                subplot(235)
                hold on
                plot(fract_change_choosingHR_aftHR_vs_LR_F_all(xi,1), fract_change_choosingHR_aftHR_vs_LR_F_all(xi,end), 'r.')
                plot([mn mx], [mn mx])
                title('HR vs LR aft HR')
                xlabel('Short ITI')
                
                subplot(236)
                hold on
                plot(fract_change_choosingLR_aftLR_vs_HR_F_all(xi,1), fract_change_choosingLR_aftLR_vs_HR_F_all(xi,end), 'r.')
                plot([mn mx], [mn mx])
                title('LR vs HR aft LR')
            end
            
        end
        
    else
        %%
        
        numSe = sum(~isnan(B_all(:,1)));
        len = length(nanmean(B_all));
        
        figure; hold on
        errorbar(1:len, nanmean(B_all), nanstd(B_all)./sqrt(numSe), 'k.')
        plot([0 len+1],[0 0], ':')
        
        %     u can only do this if binRates==0
        %     a = [stats_all.p]; % doesn't have same dimensions as B_all bc of removed sessions.
        %     nanmean(a,2)
        
        
    end
    
end

%%
%{
%% look at p values of B_ITI for s and f
    liti = (length(vec_iti)-1);
    pval_itiF = NaN(liti, length(stats_all));
    pval_itiS = NaN(liti, length(stats_all));
    for ise = 1:length(stats_all), pval_itiF(:,ise) = [stats_all(ise).p(end-liti+1 : end)]; end
    for ise = 1:length(stats_all), pval_itiS(:,ise) = [stats_all(ise).p(end-liti-liti+1 : end-liti)]; end
    
    fprintf(['average B_itiS p value for all ITIs: ', repmat('%.2f  ', [1,liti]), '\n'], nanmean(pval_itiS,2))
    fprintf(['average B_itiF p value for all ITIs: ', repmat('%.2f  ', [1,liti]), '\n'], nanmean(pval_itiF,2))
    
    
    %% plot B_iti for long vs. short iti.
    mn = min([B_itiF_all(:); B_itiS_all(:)]);
    mx = max([B_itiF_all(:); B_itiS_all(:)]);
    
    figure;
    subplot(211)
    plot(B_itiS_all(:,1), B_itiS_all(:,2), 'k.')
    hold on
    line([mn mx], [mn mx])
    title('B\_iti\_s')
    xlabel('B of short ITI')
    ylabel('B of long ITI')
    
    subplot(212)
    plot(B_itiF_all(:,1), B_itiF_all(:,2), 'k.')
    hold on
    plot([mn mx], [mn mx])
    title('B\_iti\_f')
    xlabel('B of short ITI')
    ylabel('B of long ITI')
    
%}