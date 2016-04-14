%filters data based on set lapse rate
%saves structure of pooled data

clc
clear all
close all

subject = 'am016';
thedate = '09-Apr-2013';
ndays = 45;
species = 'mice';


IncludeData = {};
counter =0;
all_data = [];
prop_chose_right = 0;
total = 0;
mean_categories = 0;
addeddates = {};
propChoice = {};

lapse_rate = 0.20;

modality = 'all'; %modality = 'aud', 'multi','vis','all'

for d = 1: ndays
    
    [all_data prop_chose_right total mean_categories] = plot_n_days_mice_kachimac2(subject, thedate, 1, species, 0, 0);
    
    
    if (~isempty(all_data))&& (~isempty(prop_chose_right)) %ignore empty datasets
        if strcmp(modality,'aud')
            all_data=all_data([all_data.visual_or_auditory] == -1);
            
        end
        if strcmp(modality,'multi')
            all_data=all_data([all_data.visual_or_auditory] == 0);
            
        end
        if strcmp(modality,'aud')
            all_data=all_data([all_data.visual_or_auditory] == 1);
            
        end
        if strcmp(modality,'all')
            
        end
        %inclusion criteria
        
        %         all_data = all_data([all_data.wait_duration] >= 1); % wait duration criteria
        %
        if (~isempty(all_data))
            
            if (prop_chose_right(1) < lapse_rate && prop_chose_right(end) > (1-lapse_rate)) %select based on lapse rate
                
                counter = counter + 1;
                IncludeData{counter} = all_data;
                
                addeddates{1,counter} = thedate;
                addeddates{2,counter} = length(all_data);
                
                propChoice{1,counter} = [mean_categories' prop_chose_right' total'];
                
            end %lapse rate selection
        end %isempty 1
    end %isempty 2
    
    thedate = datestr(datenum(thedate)-1);
    
    %     clear all_data prop_chose_right total mean_categories
    
end

%make sure all structures consistent
if ~isempty(IncludeData)
    reference =IncludeData{1,1};
    for k = 1: length(IncludeData)
        
        [newdata, refref] = make_structure_fields_consistent(IncludeData{1,k},reference);
        IncludeData{k} = newdata;
        
    end
    DataMatx = cell2mat(IncludeData);
    
    cd '~/Churchland Lab/repoland/playgrounds/Kachi/CTA/'
    %save parameters of interest
    save([subject ' ' modality '_filtered data'], 'subject','DataMatx','addeddates','lapse_rate', 'propChoice')
    
    %add plotting feature?
    
end

if isempty(IncludeData)
    %     close;
    disp(['data for ' subject ' does not satisfy criteria for inclusion'])
    cd '~/Churchland Lab/repoland/playgrounds/Kachi/CTA/'
    
end


