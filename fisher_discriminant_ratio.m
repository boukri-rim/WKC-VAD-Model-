function [FDR, idx] = fisher_discriminant_ratio(X, y)

% Class separation and the mean of each class
voiced_class = X(y == 1, :); % voiced
unvoiced_class = X(y == 0, :); % unvoiced
all_classes=X; %voiced and unvoiced frames
%The number of frames per class
n1=length(voiced_class); %number of frame in the voiced class
n2=length(unvoiced_class);%number of frame in the unvoiced class
%Calculation of class-wise means and the Fisher Discriminant Ratio for each feature
mu1 = mean(voiced_class, 1);
mu2 = mean(unvoiced_class, 1);
mu=mean(all_classes, 1);
num=n1*(mu1-mu).^2+n2*(mu2-mu).^2;
demo=sum((voiced_class(1:n1,:)-mu1).^2)+sum((unvoiced_class(1:n2,:)-mu2).^2);
FDR=num./demo;

% Sorting of features in descending order of FDR
[~, idx] = sort(FDR, 'descend');
% Output display
disp('Fisher Discriminant Ratio for each feature:');
disp(FDR);
disp('Indices of features ordered by their importance :');
disp(idx);
end
