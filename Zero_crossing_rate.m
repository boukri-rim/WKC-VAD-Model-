function[contZc,Zc]=Zero_crossing_rate(Tb,N)
e(1)=0;
    contZc = 0;%pour annuler l'accumulation de compteur 
    for j=2:N
     e(j)=abs(sign (Tb(j)) - sign (Tb(j-1)));
     if e(j) == 2
         contZc = contZc+1;
     end
    end
    e;
    contZc;
    Z=sum(e);
    Zc=(1/(2*N))*Z;%M=80 longueur de trame 160=M*2;
end
