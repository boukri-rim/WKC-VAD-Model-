function [correct,TR,FA,DCF,TP,FN,TN,Accuracy,FP,t_compression]= critical_parameter(X,y_net,dataset_correct_snr15)
correct=0;

nbr_voice=0;
nbr_silence=0;
TH=0;
TC=0;
TP=0;%the true positive
FN=0;%the false negative
TN=0;%for the true negative
nbr_silence1=0;
FA=0;%false alarme or false positive FP
for k=1:length(X)
   %/////////correct
 if y_net(k)==dataset_correct_snr15(k)
    correct=correct+1;
 end
   %/////////TP:true positif
 if (y_net(k)==1 && dataset_correct_snr15(k)==1)
    TP=TP+1;
 end
  if (y_net(k)==0 && dataset_correct_snr15(k)==0)
    TN=TN+1;
 end
 %/////////FA:FAUSSE ACCEPTANCE
 if (y_net(k)==1 && dataset_correct_snr15(k)==0)
     FA=FA+1;
 end
 if (y_net(k)==0 && dataset_correct_snr15(k)==1)
     FN=FN+1;
 end
   %/////////TR:TRUE REJECTION
 if dataset_correct_snr15(k)==0
      nbr_silence=nbr_silence+1;
 else
   TH=TH+1;%nbr de voix
 end
 if y_net(k)==0
      nbr_silence1=nbr_silence1+1;     
 else
 TC=TC+1;
 end
end
TP
TN
t_compression=(TN/length(X))*100;
correct=(correct/length(X))*100;
TP=(TP/TH)*100;
FN=(FN/TH)*100;
FA=(FA/nbr_silence)*100;
FP=FA;
TR=FN;
TN=(TN/nbr_silence)*100;
DCF=(TR*0.75)+(0.25*FA);
Accuracy=(((TP*TN)-(FP*FN))/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))*100;

end
