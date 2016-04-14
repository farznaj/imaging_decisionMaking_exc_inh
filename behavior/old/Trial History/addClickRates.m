for i = 1:numel(dat)
    
    i
    dat(i).this_aud_click_rate =size(find(cumsum((dat(i).auditoryIsis * 50)+15)<1000),2)+1;   % Because remember that the stim keeps going...  numel(dat(i).auditory_isis) + 1;
    dat(i).this_vis_click_rate =size(find(cumsum((dat(i).visualIsis * 50)+15)<1000),2)+1; 
    
    do_we_reward_right = ~isempty(dat(i).parsedEvents.states.right_on);  %~isempty(dat(i).parsed_events.states.reward);
    do_we_reward_left = ~isempty(dat(i).parsedEvents.states.left_on); 
    
    got_a_reward = ~isempty(dat(i).parsedEvents.states.reward);
    if got_a_reward == 1
        dat(i).got_a_reward = 1;
    else
        dat(i).got_a_reward = 0;
    end;
    
    
%the "correct_choice" was an unfortunate name for that structure field. 
%"correct_choice" is actually referring to the rewarded side on the given trial, not whether the animal made a correct choice.
%i.e. correct_choice = 0 means rate < CB (should go left) and correct_choice= 1 means rate > CB (should go right).
%So trial_success_history is looking for trials where the preceding trial was slow rate and the animal went left (-1) 
%or high rate and the animal correctly went right (1) -- vice versa for trial_failure_history
    
    if do_we_reward_right == 1
        dat(i).correctSide = 1;
    elseif do_we_reward_left == 1
        dat(i).correctSide = 0;
    else
        dat(i).correctSide = NaN;
    end;

    
    if dat(i).visualOrAuditory == 1 %visual
    dat(i).show_visual = 1;
    dat(i).show_audio = 0;
    elseif dat(i).visualOrAuditory == -1 %auditory
    dat(i).show_visual = 0;
    dat(i).show_audio = 1;
    elseif dat(i).visualOrAuditory == 0; %multisensory
    dat(i).show_visual = 1;
    dat(i).show_audio = 1;
    end;
    
    
end;

