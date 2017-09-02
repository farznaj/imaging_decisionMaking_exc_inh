function [day, dayLast, days2exclude] = setMouseAnalysisDays(mouse, imaging)
if ~exist('imaging','var')
    imaging = 0; % days to include when comparing results with imaging.
end

switch mouse
    
    case 'fn03'
        day = {'13-Apr-2015'}; % {'30-Mar-2015'}; % {'13-Apr-2015'} % between 03/30 and 04/13 multi-sensory was added and he had good behavior too.
        dayLast = {'09-Mar-2015'};
        days2exclude = {'25-Mar-2015'};
        
    case 'fn04'
        day = {'30-Mar-2015'}; % {'03-Apr-2015'}; % {'30-Mar-2015'}; % 3/30 last day of multi-sens w no uni-sens. 4/3: includes uni-sens too.
        dayLast = {'19-Mar-2015' };
        days2exclude = {'06-Apr-2015', '10-Apr-2015', '13-Apr-2015' , '15-Apr-2015' };
        
    case 'fn05'
        day = {'30-Mar-2015'}; % {'30-Mar-2015'}; % {'02-Apr-2015'}; % after  03/30 uni-sens.
        dayLast = {'23-Mar-2015'}; % {'23-Mar-2015'}; % {'17-Mar-2015'}; % 3/17: contingency started.
        days2exclude = {'03-Apr-2015', '06-Apr-2015' };
        
    case 'fn06'
        day = {'13-Apr-2015'}; % {'30-Mar-2015'}; % {'13-Apr-2015'};  % 4/13: if u want mult-sens
        dayLast = {'16-Mar-2015'}; % {'16-Mar-2015'}; % you can try 3/16 too.
        days2exclude = {'01-Apr-2015' , '03-Apr-2015', '06-Apr-2015', '07-Apr-2015', '09-Apr-2015', '10-Apr-2015' };
        
    case 'fni11' % not a good mouse. always biased.
        day = {'16-Jul-2015'};
        dayLast = {'06-Jun-2015'};
        days2exclude = {};
        
    case 'fni16'
%         if imaging
%             day = {'29-Oct-2015'};
%             dayLast = {'17-Aug-2015'};
%             days2exclude = {'14-Sep-2015', '15-Oct-2015'};        
%         else
            day = {'29-Oct-2015'};
            dayLast = {'30-Sep-2015'};
            days2exclude = {'15-Oct-2015'};        
%         end
    case 'fni17'
%         if imaging
%             day = {'02-Nov-2015'};
%             dayLast = {'14-Aug-2015'}; % {'02-Sep-2015'};
%             days2exclude = {'29-Sep-2015'};        
%         else
            day = {'02-Nov-2015'};
            dayLast = {'14-Sep-2015'}; % {'02-Sep-2015'};
            days2exclude = {'29-Sep-2015'};                    
%         end
    case 'fni18'
%         if imaging
%             day = {'17-Dec-2015'};
%             dayLast = {'30-Nov-2015'}; % {'07-Oct-2015'};
%             days2exclude = {'09-Nov-2015' , '11-Nov-2015' , '12-Nov-2015' , '13-Nov-2015' };
%             % days in scope rig, mouse has been doing something but w intervention and not great: Nov 16, 17, 18, 19, 20, 23, 24, 25
%         else
            day = {'17-Dec-2015'};
            dayLast = {'30-Nov-2015'}; % {'07-Oct-2015'};
            days2exclude = {'09-Nov-2015' , '11-Nov-2015' , '12-Nov-2015' , '13-Nov-2015' };
            % days in scope rig, mouse has been doing something but w intervention and not great: Nov 16, 17, 18, 19, 20, 23, 24, 25            
%         end
    case 'fni19'
%         if imaging
%             day = {'01-Nov-2015'};
%             dayLast = {'30-Sep-2015'}; % {'02-Sep-2015'};
%             days2exclude = {'14-Oct-2015', '21-Oct-2015' };
%         else
            day = {'01-Nov-2015'};
            dayLast = {'30-Sep-2015'}; % {'02-Sep-2015'};
            days2exclude = {'14-Oct-2015', '21-Oct-2015' };            
%         end        
    case 'hni01'
        day = {'21-Sep-2015'};
        dayLast = {'13-Aug-2015'};
        days2exclude = {};        
        
    case 'hni04'
        day = {'22-Sep-2015'};
        dayLast = {'11-Aug-2015'};
        days2exclude = {};                
end

