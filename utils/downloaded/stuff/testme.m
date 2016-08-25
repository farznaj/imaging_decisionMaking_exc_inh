function testme
figure('KeyPressFcn', @(src,evnt)onKeyPressRelease(evnt,'press'),...
    'KeyReleaseFcn',@(obj,evnt)onKeyPressRelease(evnt,'release')...
    );
end

function ch = onKeyPressRelease(evnt, pressRelease)
% evnt
% [strcmp(evnt.Key, 'return') || strcmp(evnt.Character, 'return')
% strcmp(evnt.Key, 'escape') || strcmp(evnt.Character, 'escape')
% strcmp(evnt.Key, '0') || strcmp(evnt.Character, '0')]

disp(evnt)
disp(pressRelease)
end