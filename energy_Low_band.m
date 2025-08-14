function[El]=energy_Low_band(R,N)
% a/ calcule matrix Toplitz
     Rt= toeplitz(R);
     % b/ calcule la réponse impulsionnelle d'un filtre FIR
        for  n = 1:13
             h(6)=1;
             h(n)= sin(2*pi*(1100/8000)*(n-6))./(pi*(n-6));
        end
        h;
     El = 10*log10((1/N)*h*Rt*h');
end