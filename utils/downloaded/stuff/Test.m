function Test()
    global T;
    T.Fig = figure;
    set(T.Fig, 'KeyPressFcn', @printchar)

function printchar(src, event)
    T.CurrentCharacter = event.Character;
    fprintf('%c', T.CurrentCharacter);
    T

% 
% ptr = 0;
% h_fig = figure;
% handles.ptr = ptr;
% guidata(h_fig, handles)
% set(h_fig, 'KeyPressFcn', @(h_obj, evt) KeyPress(h_obj, evt, varargin));
% 
%     function KeyPress(h_obj, evt, varargin)
%             handles = guidata(h_obj);
%             if(strcmp(evt.Key, 'uparrow'))
%                 handles.ptr = handles.ptr + 1
%             end 
%             guidata(h_obj, handles) % this saves the value of ptr
%     end