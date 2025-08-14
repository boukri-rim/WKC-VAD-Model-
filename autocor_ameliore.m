function[r]=autocor_ameliore(R,fs)
% c/ calcule de wlag(fenêtre de décalage d'autocorrelation)
        for k=1:10
     wlag(k)=exp(-0.5*(((2*pi*60*k)/fs)^2));
end
wlag;
% 5/ calcule les coefficiens d'autocorrelation améliorè 
     r(1)=R(1)*1.0001;
     for k=2:11
     r(k)=R(k)*wlag(k-1);
     end
     R;
     R(1);
     r;
     wlag;
     Rr=[r R(12) R(13)];
end