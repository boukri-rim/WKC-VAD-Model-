function [metrics, CM] = classification_report(ytrue, ypred)
% Accuracy, Precision, Recall, F1 + matrice de confusion (labels {0,1})
ytrue = ytrue(:); ypred = ypred(:);
TN = sum(ytrue==0 & ypred==0);
FP = sum(ytrue==0 & ypred==1);
FN = sum(ytrue==1 & ypred==0);
TP = sum(ytrue==1 & ypred==1);
acc = (TP+TN) / max(numel(ytrue),1);
prec = TP / max(TP+FP,1);
rec  = TP / max(TP+FN,1);
f1   = 2*prec*rec / max(prec+rec,eps);
metrics = struct('acc',acc,'precision',prec,'recall',rec,'f1',f1);
CM = [TN FP; FN TP];
end
