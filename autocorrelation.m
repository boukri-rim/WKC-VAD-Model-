 function [R]=autocorrelation(y,N)
% 4/ calcule les coefficiens d'Autocorrelation
    for k=1:13
     pro = zeros(1,N);%pour ne pas m�langer les valeur avec les valeur
    for n=k:N          %pr�c�dant
        pro(n)=y(n)*y(n-(k-1));
    end
    pro;
    R(k)=sum(pro);%autocorrelation
    end
R;
end