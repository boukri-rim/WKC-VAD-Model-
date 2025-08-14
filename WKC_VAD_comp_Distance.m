clear all
clc
%%%%%%%%%%%%%%%%The distance was chosen based on weak-noise data
%%%%%%%%%%%%%%%%%%%%%% LABELS NOIZEUS (train) and FDR weights
load('link\le rapport discriminant de fisher maximum correctmoy.mat') %%%% k-means weights
load('link\NOIZEUS_labels.mat');   %%%%%%%%%%% training labels
%%%%%%%%%%%%%%%%%%%%%% LABELS TIMIT (test)
load("link\TIMIT_labels.mat");%%%%%%%%%%%%% testing labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%clean audio of NOIZEUS dataset (train)
filename ='link\audio_cean.wav';[audio_clean, fs] = audioread(filename);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   NOIZEUS dataset for training
%%%%%%-----Weak Noisy-------%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SNR15
filename ='link\noisy_audio_sn15.wav';[noisy_audio_sn15, fs] = audioread(filename);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SNR10
filename ='link\noisy_audio_sn10.wav';[noisy_audio_sn10, fs] = audioread(filename);
%%%%%%-----Strong Noisy-------%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SNR5
filename ='link\noisy_audio_sn5.wav';[noisy_audio_sn5, fs] = audioread(filename);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SNR0
filename ='link\noisy_audio_sn0.wav';[noisy_audio_sn0, fs] = audioread(filename);
%%%%%%%%%%%%%%%%%%%%%% CONCAT TRAIN (clean + Weak_Noisy_audio)
training_audio=[audio_clean;noisy_audio_sn15;noisy_audio_sn10];
   [features_training]=FeatureExtractor(training_audio, fs)'; %%%% pre-processing and Feature extraction
  training_labels=[NOIZEUS_labels]';
%%%%%%%%%%%%%%%%%%%%%%  TIMIT dataset for testing
filename ='link\audio_clean.wav';
[audio_clean, fs] = audioread(filename);
%%%%%%-----Weak Noisy-------%%%%%%%%%%%%%%%%
filename ='link\noisy_audio_sn25.wav';[noisy_audio_sn25, fs] = audioread(filename);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SNR15
filename ='link\noisy_audio_sn15.wav';[noisy_audio_sn15, fs] = audioread(filename);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SNR10
filename ='link\noisy_audio_sn10.wav';[noisy_audio_sn10, fs] = audioread(filename);
%%%%%%-----Strong Noisy-------%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SNR5
filename ='link\noisy_audio_sn5.wav';[noisy_audio_sn5, fs] = audioread(filename);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SNR0
filename ='link\noisy_audio_sn0.wav';[noisy_audio_sn0, fs] = audioread(filename);
%%%%%%%%%%%%%%%%%%%%%% CONCAT TEST (clean + Weak_Noisy_audio)
 testing_audio=[audio_clean;noisy_audio_sn25;noisy_audio_sn15;noisy_audio_sn10];
 labels_testing=TIMIT_labels';
 [features_testing]=FeatureExtractor(testing_audio, fs)'; %%%% pre-processing and Feature extraction
%  % Xtrain,Xtest :Feature matrix (N_frames x N_features)
% % ytrain,ytest : Label vector (1 = voiced, 0 = unvoiced)
% % Comparative performance analysis of K-means clustering with Cityblock versus Euclidean distance
% % %% ========= 1) Cityblock (Manhattan) =========
% %% ========================= Data preparation =========================
   Xtrain=features_training(:,1:5);
ytrain=training_labels(:)';
    Xtest=features_testing(:,1:5);
ytest=labels_testing(:)';
[N,d] = size(Xtrain);
% Prepare the weights
w = prepare_feature_weights(F_correct_moy, d);
%% ========================== 1) Cityblock (L1) ==============================
out_cb = kmeans_improve(Xtrain, 2, ...
    'Distance','cityblock', ...
    'Replicates',10, ...
    'MaxIter',300, ...
    'Tol',1e-4, ...
    'Standardize',true, ...
    'FeatureWeights',w, ...
    'Labels',ytrain);
fprintf('CITYBLOCK: Inertia = %.3f | Best replicate = %d | Train Acc (inside) = %.2f%%\n', ...
        out_cb.inertia, out_cb.repBest, out_cb.acc);
% Construit un modèle complet (centroïdes + standardisation + poids + mapping)
model_cb = build_model_struct(out_cb, Xtrain, w, 'cityblock');

% Prédictions cohérentes
[~, yhat_train_cb] = kmeans_predict_consistent(model_cb, Xtrain);
[~, yhat_test_cb] = kmeans_predict_consistent(model_cb, Xtest);
 [yhat_train_cb]= smoothing(Xtrain,yhat_train_cb);
  [yhat_test_cb]= smoothing(Xtest,yhat_test_cb);
[mtr_cb, cmtr_cb] = classification_report(ytrain, yhat_train_cb);
[mte_cb, cmte_cb] = classification_report(ytest, yhat_test_cb);

fprintf('\n=== CITYBLOCK (Manhattan) ===\n');
fprintf('TRAIN: Acc=%.2f%%  Prec=%.3f  Rec=%.3f  F1=%.3f\n', ...
        100*mtr_cb.acc, mtr_cb.precision, mtr_cb.recall, mtr_cb.f1);
fprintf('TEST : Acc=%.2f%%  Prec=%.3f  Rec=%.3f  F1=%.3f\n', ...
        100*mte_cb.acc, mte_cb.precision, mte_cb.recall, mte_cb.f1);
%% ==================== 2) Euclidean (sqeuclidean) ===========================
out_eu = kmeans_improve(Xtrain, 2, ...
    'Distance','sqeuclidean', ...
    'Replicates',10, ...
    'MaxIter',300, ...
    'Tol',1e-5, ...
    'Standardize',true, ...
    'FeatureWeights',w, ...
    'Labels',ytrain);
fprintf('EUCLIDEAN: Inertia = %.3f | Best replicate = %d | Train Acc (inside) = %.2f%%\n', ...
        out_eu.inertia, out_eu.repBest, out_eu.acc);
model_eu = build_model_struct(out_eu, Xtrain, w, 'sqeuclidean');

[~, yhat_train_eu] = kmeans_predict_consistent(model_eu, Xtrain);
[~, yhat_test_eu] = kmeans_predict_consistent(model_eu, Xtest);
 [yhat_train_eu]= smoothing(Xtrain,yhat_train_eu)';
  [yhat_test_eu]= smoothing(Xtest,yhat_test_eu)';
[mtr_eu, cmtr_eu] = classification_report(ytrain, yhat_train_eu);
[mte_eu, cmte_eu] = classification_report(ytest,  yhat_test_eu);

fprintf('\n=== EUCLIDEAN (sqeuclidean) ===\n');
fprintf('TRAIN: Acc=%.2f%%  Prec=%.3f  Rec=%.3f  F1=%.3f\n', ...
        100*mtr_eu.acc, mtr_eu.precision, mtr_eu.recall, mtr_eu.f1);
fprintf('TEST : Acc=%.2f%%  Prec=%.3f  Rec=%.3f  F1=%.3f\n', ...
        100*mte_eu.acc, mte_eu.precision, mte_eu.recall, mte_eu.f1);

%% ======================== 3) Select Distance & Save ============================
if mte_cb.acc > mte_eu.acc
    bestModel = model_cb; bestName = 'Cityblock'; bestMetrics = mte_cb;
else
    bestModel = model_eu; bestName = 'Euclidean'; bestMetrics = mte_eu;
end

save('kmeans_model_wkcvad.mat','-struct','bestModel');

fprintf('\nBest on TEST: %s | Acc=%.2f%%  Prec=%.3f  Rec=%.3f  F1=%.3f\n', ...
        bestName, 100*bestMetrics.acc, bestMetrics.precision, bestMetrics.recall, bestMetrics.f1);
fprintf('Model (struct) saved to kmeans_model_wkcvad.mat\n');

try
    figure; confusionchart(ytrain, yhat_train_cb, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
    title('Confusion Matrix — TRAIN (Cityblock)');
    figure; confusionchart(ytest, yhat_test_cb, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
    title('Confusion Matrix — TEST (Cityblock)');
    figure; confusionchart(ytrain, yhat_train_eu, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
    title('Confusion Matrix — TRAIN (Euclidean)');
    figure; confusionchart(ytest, yhat_test_eu, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
    title('Confusion Matrix — TEST (Euclidean)');
catch
    disp('Confusion matrix TRAIN [TN FP; FN TP] (Cityblock):'); disp(cmtr_cb);
    disp('Confusion matrix TEST  [TN FP; FN TP] (Cityblock):');  disp(cmte_cb);
    disp('Confusion matrix TRAIN [TN FP; FN TP] (Euclidean):');  disp(cmtr_eu);
    disp('Confusion matrix TEST  [TN FP; FN TP] (Euclidean):');   disp(cmte_eu);
end
