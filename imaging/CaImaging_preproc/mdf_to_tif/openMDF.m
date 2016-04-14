function [mfile, OpenResult] = openMDF(mdfFileName)
% Uses MCSX library that comes with MView to open a MDF file in matlab.

if ishandle(1), close(1), end

fhandle = figure(1);
mfile = actxcontrol('MCSX.Data', [0, 0, 500, 500], fhandle);
OpenResult = mfile.invoke('OpenMCSFile', mdfFileName);
% openresult = mfile.OpenMCSFile(fulldir)

end

