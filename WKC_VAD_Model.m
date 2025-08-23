%% ========================================================================
%Complete Program(ordered)
%1) K-means:comparison 'Manhattan ' vs 'euclidean' (Replicates=10)
%2)Selection of the best model+ test(accuracy before adjustment)

%(A) extract features ========================================================================
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
   Xtrain=features_training;
  training_labels=[NOIZEUS_labels]';
  ytrain=training_labels;
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
Xtrain = Xtrain(:,1:5);                            % garder 5 features
[~, d]    = size(Xtrain);

% === Labels train (8 segments: 1 clean + 7 bruits) ===
labels_train_mat = [D_sp1_30_clean, repmat(D_sp1_30_clean, 1, 7)];
ytrain = labels_train_mat(:);                         % N x 1
Ntr  = size(Xtrain,1);
Nlab = numel(ytrain);
Nfit = min(Ntr, Nlab);
Xtrain = Xtrain(1:Nfit,:);
ytrain = ytrain(1:Nfit);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%  TIMIT dataset for testing
load("C:\Users\Admin\Desktop\article1\data\TIMIT\data_sa_160\data_sa1_120.mat"); % data_sa1_120
filename ='C:\Users\Admin\Desktop\article1\data\TIMIT\noise\sa1_120_awgn_sn25.wav';   [sa1_120_awgn_sn25, fs]  = audioread(filename);
filename ='C:\Users\Admin\Desktop\article1\data\TIMIT\noise\sa1_120_babble_sn25.wav'; [sa1_120_babble_sn25, fs]= audioread(filename);
filename ='C:\Users\Admin\Desktop\article1\data\TIMIT\noise\sa1_120_car_sn25.wav';    [sa1_120_car_sn25, fs]   = audioread(filename);
filename ='C:\Users\Admin\Desktop\article1\data\TIMIT\noise\sa1_120_awgn_sn15.wav';   [sa1_120_awgn_sn15, fs]  = audioread(filename);
filename ='C:\Users\Admin\Desktop\article1\data\TIMIT\noise\sa1_120_babble_sn15.wav'; [sa1_120_babble_sn15, fs]= audioread(filename);
filename ='C:\Users\Admin\Desktop\article1\data\TIMIT\noise\sa1_120_car_sn15.wav';    [sa1_120_car_sn15, fs]   = audioread(filename);

testing_audio = [sa1_120_awgn_sn25; sa1_120_babble_sn25; sa1_120_car_sn25; ...
                 sa1_120_awgn_sn15; sa1_120_babble_sn15; sa1_120_car_sn15];

% Labels test (6 concatenated segments)
labels_testing = repmat(data_sa1_120(:), 6, 1);
ytest=labels_testing;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% K-Means Training (cityblock)
w   = prepare_feature_weights(F_correct_moy, d);         % 1 x d (ou [])
 [features_testing]=FeatureExtractor(testing_audio, fs)'; %%%% pre-processing and Feature extraction
 Xtest=features_testing;
 Xtest = Xtest(:,1:5);                            % select 5 features


%% (B) ---------------- 1) K-MEANS : cityblock vs euclidean -----------------
Replicates = 10;   % "the number of tests were conducted is 10"
fprintf('\n=== (1)  K-means Training : cityblock vs euclidean (Replicates=%d) ===\n', Replicates);

out_cb = kmeans_improve(Xtrain, 2, ...
    'Distance','cityblock', 'Replicates',Replicates, ...
    'MaxIter',300, 'Tol',1e-5, 'Standardize',true, ...
    'FeatureWeights', w, 'Labels', ytrain);

out_eu = kmeans_improve(Xtrain, 2, ...
    'Distance','euclidean', 'Replicates',Replicates, ...
    'MaxIter',300, 'Tol',1e-5, 'Standardize',true, ...
    'FeatureWeights', w, 'Labels', ytrain);

fprintf('  cityblock : Inertia = %.4g | BestRep = %d | Acc(train)=%.2f%%\n', out_cb.inertia, out_cb.repBest, out_cb.acc);
fprintf('  euclidean : Inertia = %.4g | BestRep = %d | Acc(train)=%.2f%%\n', out_eu.inertia, out_eu.repBest, out_eu.acc);

%% (C) -------- 2) Final Model Selection + Pre-Update Testing --------
% Choice:best accuracy (Otherwise :lowest inertia)
choose_cb = false;
if ~isempty(out_cb.acc) && ~isempty(out_eu.acc)
    if out_cb.acc > out_eu.acc
        choose_cb = true;
    elseif out_cb.acc == out_eu.acc
        choose_cb = (out_cb.inertia <= out_eu.inertia);
    end
else
    choose_cb = (out_cb.inertia <= out_eu.inertia);
end

if choose_cb
    out_best  = out_cb;
    dist_best = 'cityblock';
else
    out_best  = out_eu;
    dist_best = 'euclidean';
end
fprintf('\n=== (2) Selected Model: %s ===\n', dist_best);

% Prediction-compatible Model
model = build_model_struct(out_best, Xtrain, w, dist_best);
model.nk = accumarray(out_best.idx, 1, [size(out_best.C,1), 1], @sum, 0);

% Test Accuracy Before Update
[~, yhat_before] = kmeans_predict_consistent(model, Xtest);
 [yhat_before]= smoothing(Xtest,yhat_before);
acc_before = mean(yhat_before == ytest);
fprintf('Test Accuracy Before Update = %.2f%% (N=%d)\n', 100*acc_before, numel(ytest));



% %% ============================ Local Functions ============================
% 
function model = build_model_struct(out, Xref, w, distName)
    mu = mean(Xref,1);
    sg = std(Xref,[],1); sg(sg==0)=1;
    model = struct();
    model.C    = out.C;     % centroids in the standardized/ weighted space
    model.map  = out.map;   % mapping cluster -> (if labels are provided during training)
    model.mu   = mu;        % standardization
    model.sg   = sg;
    if ~isempty(w), model.w = w(:).'; else, model.w = []; end
    model.dist = distName;  % 'cityblock' | 'euclidean' | 'sqeuclidean'
end

function [cl, yhat] = kmeans_predict_consistent(model, X)
    Xz = (X - model.mu)./model.sg;
    if ~isempty(model.w), Xz = Xz .* model.w; end
    D = pdist2_compat(Xz, model.C, model.dist);
    [~, cl] = min(D, [], 2);
    if isfield(model,'map') && ~isempty(model.map)
        yhat = model.map(cl);
    else
        yhat = cl;
    end
end

function D = pdist2_compat(X, C, distName)
    if nargin<3, distName='cityblock'; end
    distName = lower(distName);
    [N,~] = size(X); K = size(C,1);
    D = zeros(N,K);
    switch distName
        case 'cityblock'
            for k = 1:K, D(:,k) = sum(abs(X - C(k,:)), 2); end
        case 'sqeuclidean'
            for k = 1:K, D(:,k) = sum((X - C(k,:)).^2, 2); end
        case 'euclidean'
            for k = 1:K, D(:,k) = sqrt(sum((X - C(k,:)).^2, 2)); end
        otherwise
            for k = 1:K, D(:,k) = sum(abs(X - C(k,:)), 2); end
    end
end