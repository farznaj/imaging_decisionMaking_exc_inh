% fig_h = figure;
% set(fig_h,'KeyPressFcn', @key_pressed_fcn);
function key_pressed_fcn(fig_obj,eventDat)

% ch = {eventDat.Key, eventDat.Character}

get(fig_obj, 'CurrentKey')
get(fig_obj, 'CurrentCharacter')
get(fig_obj, 'CurrentModifier')

% or

disp(eventDat)
