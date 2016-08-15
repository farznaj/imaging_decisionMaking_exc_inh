
function FiltSignal=gaussFilter(SD, inSignal)
flgRawSig=0;
if size(inSignal,1)< size(inSignal,2) % make sure it is column vector
    inSignal=inSignal';
    flgRawSig=1;
end
filtwidth=8*SD; % filter width
b=fspecial('gaussian',[1,filtwidth],SD); % generate filter coefficients
PadSignal=[repmat(mean(inSignal(1:SD,:),1), floor(filtwidth/2),1); inSignal;repmat(mean(inSignal(end-SD:end,:),1), floor(filtwidth/2),1)]; % padd signals with mean values at start and the end to handle filter shift
FiltSignal=filter(b,1,PadSignal); % filter the signal
FiltSignal=FiltSignal(2*floor(filtwidth/2):length(PadSignal)-1,:); % crop the signal to the right size of the input signal

if flgRawSig
  FiltSignal=FiltSignal'; % preserve signal orientation if raw signal
end
end