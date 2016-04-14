function signal_visual = create_visual_signal(isis, rate, brightness, flash_duration, long_isi, short_isi, offset)

timevec = (0 : 1/rate : flash_duration);
freq = 200*brightness; % we can't detect a 300 Hz flicker %FN: the freq must be high enough such that flicker fusion happens. The higher the frequency, the brighter the LED signal will be.
sine_wave = sin(2*pi * freq * timevec);

[pks,locs] = findpeaks(sine_wave);

flash = zeros(size(sine_wave));
flash(locs) = 1;
offset_dark = zeros(1, round(rate * offset));

long_dark = zeros(1, round(rate * long_isi));
short_dark = zeros(1, round(rate * short_isi));

signal_visual = [offset_dark flash];

for i = isis
    if i == 1
        signal_visual = [signal_visual short_dark flash];
    elseif i == 2
        signal_visual = [signal_visual long_dark flash];
    end
end

end


%% find inter-event intervals
a = [diff(vis_stim) 1];
ieis = diff(find(a==1))-evdur;
ieis = [find(vis_stim==1, 1)-1 ieis]

isequal(sum(ieis)+evdur*(length(ieis)-1), length(vis_stim)) % sanity check; it must be equal to 1.





