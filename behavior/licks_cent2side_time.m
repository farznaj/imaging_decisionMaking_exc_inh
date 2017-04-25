% Compute time between center lick off to side lick on 
% This is to get to know how long it takes to move the tongue from center
% to side; ie how much in advance to detecting side lick, mouse moved her
% tongue from center to side.
%
% Assumption: min(sideLick-centerLick) is perhaps when mouse kept licking
% on the center and then immediately licked on the side; so it gives a
% rough estimate of how long it takes to move the tongue from center to side.

c2sTime = nan(1, length(all_data));
for tr = 1:length(all_data)
    if ~isempty(all_data(tr).parsedEvents)
        centerLicksOff = all_data(tr).parsedEvents.pokes.C(:,2);
        if ~isempty(centerLicksOff)
            sideLicksOn = cat(1, all_data(tr).parsedEvents.pokes.L(:,1) , all_data(tr).parsedEvents.pokes.R(:,1));

            if ~isempty(sideLicksOn)
                firstSLaftCL_i = find((sideLicksOn > centerLicksOff(1))==1, 1); % index of first side lick that happens after first center lick

                if ~isempty(firstSLaftCL_i)
                    clBefSl = centerLicksOff(find((sideLicksOn(firstSLaftCL_i) < centerLicksOff)==0, 1, 'last')); % last center lick that happened before the side lick
                    c2sTime(tr) = sideLicksOn(firstSLaftCL_i)*1000 - clBefSl*1000;
                end
            end
        end
    end
end

s = sort(c2sTime); s(1:min(10,length(s)))
% figure; 
hist(c2sTime,100)

