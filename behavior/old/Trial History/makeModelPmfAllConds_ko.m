

colors = [0 1 0;0 0 0;0 0 1];



figure;
% set(gcf,'position',[1000 1067 1244 431])
% for ConditionCount = 1:3
    ConditionCount=2;
    actualChoice = [];
    modelPredRates = [];
    theseRates = [];
    
    choice_data_right_modality_ind = find(choice_data_matrix(:,ConditionCount) == 1);
    
    %take only the visual trials;
    choice_model_response_predictions_right_modality = choice_model_response_predictions(choice_data_right_modality_ind);
    Y_right_modality = Y( choice_data_right_modality_ind);
    
    choice_data_right_modality = choice_data_matrix(choice_data_right_modality_ind,:);
    
    theseRates = unique(choice_data_right_modality(:,ConditionCount+7));
    
    for i_rate = 1:length(theseRates)
        
        
        right_trials = find(choice_data_right_modality(:,ConditionCount+7) == theseRates(i_rate));
        
        modelPredRates(i_rate) = mean(choice_model_response_predictions_right_modality(right_trials));
        rightResp = Y_right_modality(right_trials);
        
        actualChoice(i_rate) = size(find(rightResp == 1),1)/size(find(rightResp == 1 | rightResp == 0),1);
    end;
    
%     subplot(1,3,ConditionCount)
    
    
    plot(theseRates,modelPredRates,'k-','color',colors(ConditionCount,:)); hold on;
    plot(theseRates,actualChoice,'ko','color',colors(ConditionCount,:));
    axprefs(gca);
    xlabel('rate');
    ylabel('Prop choice right');
    set(gca,'ylim',[0 1]);
    text(2,0.2,sprintf('bias=%0.02f',coefficient_vec(ConditionCount)));
        
        
 
    if ConditionCount == 2
        title(sprintf('%s: bs=%0.2f,+/- %0.02f, bf=%0.02f +/- %0.02f, numtrials = %0.2f',mousename,coefficient_vec(4),model_stats.se(4),coefficient_vec(6),model_stats.se(6), numtrials),'fontweight','bold','fontsize',14)
    end;
    
% end;

% cd '~/Churchland Lab/Dropbox/Kachi/trial history data/'
% saveas(gcf,[mousename 'trial history'],'jpg')



