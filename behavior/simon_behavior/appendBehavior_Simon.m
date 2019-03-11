function bhv = appendBehavior(bhv,data)
% Function to collect data from behavioral files into one unified array bhv.
% Usage: bhv = appendBehavior(bhv,data)

%% get fieldnames
if isempty(bhv)
    bFields = {};
else
    bFields = fieldnames(bhv);
end
dFields = fieldnames(data);
dFields(strcmpi(dFields,'Settings')) = [];

%% cycle trough fields and add current data to bhv structure. Create new bhv entries if required.
structCases = {'RawEvents' 'RawData'}; %cases at which substructs are present. Go one level depper to correctly append them together.
for iFields = 1:size(dFields,1)
    if isstruct(data.(dFields{iFields}))
        if ismember(dFields{iFields},bFields) %existing field
            if any(strcmpi((dFields{iFields}),structCases))
                try
                    tFields = fieldnames(bhv.(dFields{iFields}));
                    for x = 1:length(tFields)
                        bhv.(dFields{iFields}).(tFields{x}) = [bhv.(dFields{iFields}).(tFields{x}) data.(dFields{iFields}).(tFields{x})];
                    end
                catch
                    bhv.(dFields{iFields}) = [bhv.(dFields{iFields}) data.(dFields{iFields})];
                end
            else
                bhv.(dFields{iFields}) = [bhv.(dFields{iFields}) data.(dFields{iFields})];
            end
        else %new field in data
            bhv.(dFields{iFields}){1} = data.(dFields{iFields});
        end
    else
        if length(data.(dFields{iFields})) > 1 %vector or matrix
            if ischar(data.(dFields{iFields})) % carry strings in a cell container
                temp = {data.(dFields{iFields})};
            else
                if any(size(data.(dFields{iFields})) == 1) && length(data.(dFields{iFields})) >= sum(data.nTrials) %check if vector with at least nTrials entries
                    temp = data.(dFields{iFields})(1:sum(data.nTrials)); %keep nTrials entries
                else
                    if iscell(data.(dFields{iFields}))
                        temp = {data.(dFields{iFields})}; %cells are usually settings. keep in one cell per data file.
                    else
                        temp = data.(dFields{iFields}); %use all values and append together into larger matrix.
                    end
                end
            end
        else %single value
            temp = data.(dFields{iFields});
        end
        
        if ~isobject(temp) % don't append objects into larger structure
            if size(temp,2) == 1; temp = temp'; end %column vectors are transposed to rows
            
            if ismember(dFields{iFields},bFields) %existing field
                bhv.(dFields{iFields}) = [bhv.(dFields{iFields}) temp];
            else %new field in data
                bhv.(dFields{iFields}) = temp;
            end
        end
    end 
end
    

