%% AR(1)

g = 0.95;
s = [1,zeros(1,49)];
dt = 0.05;
c = filter(1,[1,-g],s);
tau = -dt/log(g);
sf = 10;
dt2 = dt/sf;
t = (0:49)*dt;
t2 = t(1):dt2:t(end);
h = exp(-t2/tau);
figure;plot(t,c,'o'); hold all; plot(t2,h,'.');


%% AR(2)
s = [0,1,zeros(1,48)];
gr = [0.6,0.95];
c2 = filter(1,poly(gr),s);
tau = -dt./log(gr);
h2 = exp(-t2/max(tau)) - exp(-t2/min(tau));
figure;plot(t,c2*abs(diff(gr)),'o'); hold all; plot(t2,h2);
