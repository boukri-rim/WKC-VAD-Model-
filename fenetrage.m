function [SW]=fenetrage(y,N)
% a/ calcule fenêtre Hamming and tracé fenetre de hamming
if N<=200
for n=1:N
    WLP(n)= 0.54-0.46*cos((2*pi*(n-1))/399);
end
else
for n = 201:240
    WLP(n) = cos(2*pi*(n-201)/159);
end
end
WLP;
 % 3/ fenetre(3trames) * Hamming 
    for o=1:N
    SW(o) = y(:,o)*WLP(:,o);
    end
    SW;
end
 