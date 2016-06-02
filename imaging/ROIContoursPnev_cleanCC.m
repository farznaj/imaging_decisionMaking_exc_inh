function CC = ROIContoursPnev_cleanCC(CC)
% In CC set the column of metadata to NaN, so plotting contours would be easy.

%{
a = [cellfun(@(x)x(1,1), CC), cellfun(@(x)size(x,2)-1, CC)];
% [a diff(a,[],2)]
eftROISeveralReg = find(diff(a,[],2)~=0); % These are Eft ROIs which have >1 contour region.
% length(eftROISeveralReg)
%}

for rr = 1:length(CC)    
    regI = 1;
    while regI < size(CC{rr}, 2)
        nElem = CC{rr}(1, regI);
        CC{rr}(:, regI) = NaN;
        regI = regI+nElem+1;
    end
end


