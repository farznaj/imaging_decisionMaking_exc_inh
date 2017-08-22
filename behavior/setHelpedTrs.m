function all_data = setHelpedTrs(all_data, defaultHelpedTrs, saveHelpedTrs, alldata_fileNam, helpedInit, helpedChoice)
% all_data = setHelpedTrs(all_data, defaultHelpedTrs, saveHelpedTrs, alldata_fileNam, helpedInit, helpedChoice);
%
% It will add fields helpedInit and helpedChoice to all_data if they
% already don't exist.
%
% if defaultHelpedTrs is true, helpedInit and helpedChoice will be set to
% false for all trials.
% 
% if saveHelpedTrs is true, all_data, including the extra helped fields,
% will be overwritten. You need to provide the mat file name containing
% all_data.
%
% helpedInit and helpedChoice are optional inputs.


%%
if logical(defaultHelpedTrs)
    [all_data.helpedInit] = deal(false);
    [all_data.helpedChoice] = deal(false);

else
    
    if ~all([isfield(all_data,'helpedInit'), isfield(all_data,'helpedChoice')])

%         disp('you need to provide helpedInit and helpedChoice trials.');
        if ~exist('helpedInit','var')
            helpedInit = input('trials helped w init?');
        end
        if ~exist('helpedChoice','var')
            helpedChoice = input('trials helped w choice?');
        end
        
        %%
        h_init = num2cell(false(1, length(all_data)));
        h_init(helpedInit) = {true};
        [all_data.helpedInit] = deal(h_init{:});
        
        h_choice = num2cell(false(1, length(all_data)));
        h_choice(helpedChoice) = {true};
        [all_data.helpedChoice] = deal(h_choice{:});
        
        clear helpedInit helpedChoice h_init h_choice
        
        %%        
%         figure(120); plot([all_data.helpedInit])
%         hold on; plot([all_data.helpedChoice])
        
        if logical(saveHelpedTrs)
            cprintf('r', 'Overwriting all_data after adding helped trials!\n')
            save(alldata_fileNam, 'all_data', '-append')
        end
    else
        fprintf('Fields helpedInit and helpedChoice already exist!\n')
    end
end



