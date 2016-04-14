function [structure1,structure2] = make_structure_fields_consistent(structure1,structure2);
% Written by JPS, 2011

structure1_fields = fieldnames(structure1);
structure2_fields = fieldnames(structure2);
[structure2_missing_fields,structure2_missing_field_indices] = setdiff(structure1_fields,structure2_fields);
[structure1_missing_fields,structure1_missing_field_indices] = setdiff(structure2_fields,structure1_fields);
missing_fields = {structure2_missing_fields{:},structure1_missing_fields{:}};
all_missing_field_indices(:,1) = [structure2_missing_field_indices(:);structure1_missing_field_indices(:)];
all_missing_field_indices(:,2) = [2*ones(numel(structure2_missing_field_indices),1);ones(numel(structure1_missing_field_indices),1)];
for MissingFieldCount = 1:numel(missing_fields)
    missing_field = missing_fields{MissingFieldCount};
    if all_missing_field_indices(MissingFieldCount,2) == 2
        %['[structure2.' missing_field '] = deal(NaN);']
        eval(['[structure2.' missing_field '] = deal(NaN);']);
    elseif all_missing_field_indices(MissingFieldCount,2) == 1
        %['[structure1.' missing_field '] = deal(NaN);']
        eval(['[structure1.' missing_field '] = deal(NaN);']);
    end
end
structure1 = orderfields(structure1);
structure2 = orderfields(structure2);
end % EOF