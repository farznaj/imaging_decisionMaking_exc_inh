a = fread(fid, 520, 'uint8=>uint8');
figure; plot(a)

%%
format long

x1 = 0;
x2 = x1+4;
% x1+1 : x2
a(x1+1 : x2)
typecast(a(x1+1 : x2), 'int32')
pause


x1 = x2;
x2 = x1+4;
% x1+1 : x2
a(x1+1 : x2)
typecast(a(x1+1 : x2), 'int32')
pause


x1 = x2;
x2 = x1+8;
% x1+1 : x2
a(x1+1 : x2)
typecast(a(x1+1 : x2), 'double')
pause


for l=1:8
    x1 = x2;
    x2 = x1+4;
    % x1+1 : x2
    a(x1+1 : x2)
    typecast(a(x1+1 : x2), 'uint32')
    pause
end


for l=1:8
    x1 = x2;
    x2 = x1+8;
    % x1+1 : x2
    a(x1+1 : x2)
    typecast(a(x1+1 : x2), 'double')
    pause
end

x1 = x2

