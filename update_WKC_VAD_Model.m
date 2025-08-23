%% ================================================================% Ordered Program
%1) Train K-means by comparing 'Manhattan' vs 'euclidean'
%2) Select the best WKC-VAD model + test (accuracy Before adjustment)
%3) Automaticaly adjust the centroids according to Dmin with the rule:
%       S(x) = | ||x - C1||_1 - ||x - C2||_1 |  >  Dmin   -> UPDATE
%     - Affiche les centroïdes après CHAQUE ajustement
%     - Compte #trames utilisées pour l’ajustement et #rejetées
% -Display the centroids after each adjustment 
% - Count the number of frames used for the adjustment and the number discarded
% ================================================================
%(A) extract features ========================================================================
clear all
clc

%%%%%%%%%%%%%%%%%%%%%% LABELS NOIZEUS (train) and FDR weights
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

out_cb = kmeans_improve(X_low_SNR, 2, ...
    'Distance','cityblock', 'Replicates',Replicates, ...
    'MaxIter',300, 'Tol',1e-5, 'Standardize',true, ...
    'FeatureWeights', w, 'Labels', y_low_SNR);

out_eu = kmeans_improve(X_low_SNR, 2, ...
    'Distance','euclidean', 'Replicates',Replicates, ...
    'MaxIter',300, 'Tol',1e-5, 'Standardize',true, ...
    'FeatureWeights', w, 'Labels', y_low_SNR);

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
model = build_model_struct(out_best, X_low_SNR, w, dist_best);
model.nk = accumarray(out_best.idx, 1, [size(out_best.C,1), 1], @sum, 0);

% Test Accuracy Before Update
[~, yhat_before] = kmeans_predict_consistent(model, Xtest);
 [yhat_before]= smoothing(Xtest,yhat_before);
acc_before = mean(yhat_before == ytest);
fprintf('Test Accuracy Before Update = %.2f%% (N=%d)\n', 100*acc_before, numel(ytest));

%% (D) ---- 3) Automatic adjustment based on Dmin+ Display at each step----
% Dmin =minimum distance between points from different classes
fprintf('\n=== (3) Dmin (cityblock) & Ajustement si | |Xi-C1|_1 - |Xi-C2|_1 | > Dmin ===\n');

if exist('min_interclass_distance_big','file') == 2
    [Dmin_cb, ~, ~] = min_interclass_distance_big( ...
        X_low_SNR, y_low_SNR, 1, 0, ...
        'Distance','cityblock', 'Standardize',true, 'FeatureWeights', w, ...
        'Method','knn', 'QueryBlock', 20000);
else
    Dmin_cb = compute_dmin_simple_cityblock(X_low_SNR, y_low_SNR, true, w);
end
fprintf('Dmin (cityblock) = %.4g\n', Dmin_cb);

% Rule-based adjustment : S(x)=| ||x-C1||_1 - ||x-C2||_1 | > Dmin -> update
[model_after, cl_after, yhat_after, logu] = kmeans_update_cityblock_margin_Dmin( ...
    model, Xtest, Dmin_cb, 'Eta', 0.05, 'MaxUpdates', inf);

% Display of the centroids after each update
T = size(logu.C_hist,3);
for t = 1:T
    n = logu.step(t);
    fprintf('  Update #%d the frame %d | S=%.4g > Dmin=%.4g\n', t, n, logu.S(n), Dmin_cb);
    disp(logu.C_hist(:,:,t)); % centroïdes après l’update t
end

% Statistics of Accepted vs Discarded Frames
nb_used   = sum(logu.didUpdate);
nb_reject = numel(logu.didUpdate) - nb_used;
fprintf('Total updates = %d | frames used (ajustement) = %d | Rejected = %d\n', T, nb_used, nb_reject);

% Accuracy TEST after update
acc_after = mean(yhat_after == ytest);
fprintf('Accuracy TEST after update  = %.2f%%\n', 100*acc_after);

% Number of tests (réplications K-means)
fprintf('Number of tests ( K-means replications) performed = %d\n', Replicates);

%% (E) ---- 4) RELATIVE DISTANCES to the final centroid + CURVES ----
C_final = model_after.C;                                  % final centroids 
[Dabs, Drel] = centroid_relative_to_final(logu, C_final, 'cityblock');
Drel_moy=(Drel(:,1)+Drel(:,2))/2;
if isempty(Dabs)
    fprintf('\nNo adjustment was performed -> no intermidiate distances..\n');
else
    fprintf('\n=== Distances with final centroids (cityblock) ===\n');
    fprintf('t\tframe\t|C1_abs\t\tC1_rel(%%)\t|C2_abs\t\tC2_rel(%%)\n');
    for t = 1:size(Dabs,1)
        fr = logu.step(t);
        fprintf('%d\t%d\t|%.4g\t\t%.2f\t\t|%.4g\t\t%.2f\n', ...
            t, fr, Dabs(t,1), 100*Drel(t,1), Dabs(t,2), 100*Drel(t,2));
    end

    % Curves (axe = #update)
    t_axis = 1:size(Drel,1);
    figure;
    plot(t_axis, 100*Drel(:,1), '-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
    plot(t_axis, 100*Drel(:,2), '-s', 'LineWidth', 1.5, 'MarkerSize', 4);
    grid on; xlabel('Ajustement step'); ylabel('relative Distance (%)');
    title('Convergence of the centroids towards C_{final} (axe = #update)');
    legend('Centroïde C_1', 'Centroïde C_2', 'Location', 'northeast');

    % Corves (axe = frame index where the update occurred)
    figure;
    plot( 100*Drel_moy,t_axis, '-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
    grid on; xlabel('relative Distance (%)'); ylabel('Frame Index (update)');
    title('Convergence of the centroids towards C_{final} (axe = trame)');
    legend('Centroïde C_1', 'Centroïde C_2', 'Location', 'northeast');
end

%% ============================ Local FONCTIONS ============================

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

function Dmin = compute_dmin_simple_cityblock(X, y, doStd, w)
    % Fallback (O(N^2)) : Dmin between frames of differrent classes in L1 (cityblock)
    if doStd
        mu = mean(X,1);
        sg = std(X,[],1); sg(sg==0)=1;
        Xz = (X - mu)./sg;
    else
        Xz = X;
    end
    if ~isempty(w), Xz = Xz .* w(:).'; end
    A = Xz(y==0,:); B = Xz(y==1,:);
    Dmin = inf;
    for i=1:size(A,1)
        d = sum(abs(B - A(i,:)), 2);
        m = min(d);
        if m < Dmin, Dmin = m; end
    end
end

% ====== AJUSTEMENT : S(x)=| ||x-C1||_1 - ||x-C2||_1 | > Dmin -> update ======
function [model, cl, yhat, logu] = kmeans_update_cityblock_margin_Dmin(model, X, Dmin, varargin)
    p = inputParser;
    addParameter(p,'Eta',0.05);
    addParameter(p,'MaxUpdates',inf);
    parse(p, varargin{:});
    eta        = p.Results.Eta;
    maxUpdates = p.Results.MaxUpdates;

    if size(model.C,1) ~= 2, error('This rule assumes K=2 centroïdes.'); end
    if ~isscalar(Dmin), error('Dmin must be a scalar.'); end

    % Standardization/same weighting as used during training
    Xz = (X - model.mu)./model.sg;
    if isfield(model,'w') && ~isempty(model.w), Xz = Xz .* model.w; end

    [N,d] = size(Xz); K = size(model.C,1);
    logu.S = zeros(N,1);
    logu.didUpdate = false(N,1);
    logu.C_hist = zeros(K,d,0);
    logu.step   = zeros(0,1);

    cl = zeros(N,1);
    haveMap = isfield(model,'map') && ~isempty(model.map);
    if haveMap, yhat=zeros(N,1); else, yhat=[]; end

    nUpdates = 0;

    for n = 1:N
        xz = Xz(n,:);

        % L1 Distances to C1 and C2
        d1 = sum(abs(xz - model.C(1,:)));
        d2 = sum(abs(xz - model.C(2,:)));

        % nearest centroid
        k_win = 1 + (d2 < d1);

        % Marge L1
        S = abs(d1 - d2);
        logu.S(n) = S;
        cl(n) = k_win;

        % Condition : S(x) > Dmin  -> update
        cond = (S > Dmin);

        if cond && (nUpdates < maxUpdates)
            if isfield(model,'nk') && ~isempty(model.nk)
                % Winning centroid (Exact incremental mean)
                nk_old = model.nk(k_win); nk_new = nk_old + 1;
                model.C(k_win,:) = (nk_old * model.C(k_win,:) + xz) / nk_new;
                model.nk(k_win)  = nk_new;
            else
                model.C(k_win,:) = (1 - eta) * model.C(k_win,:) + eta * xz;
            end
            nUpdates = nUpdates + 1;
            logu.didUpdate(n) = true;

            logu.C_hist(:,:,end+1) = model.C; 
            logu.step(end+1,1)     = n;       
        end
    end

    if haveMap
        yhat = model.map(cl);
    else
        yhat = cl;
    end
end

% ====== Winning centroid (relative distance of the centroids to the final center) ======
function [Dabs, Drel] = centroid_relative_to_final(logu, C_final, distName)
    T = size(logu.C_hist, 3);
    K = size(C_final, 1);
    if T == 0
        Dabs = zeros(0,K); Drel = zeros(0,K);
        return;
    end

    Dabs = zeros(T, K);
    Drel = zeros(T, K);

    % Centroid-based normalization using the distance from the first to the final snapshot
    D0 = zeros(1,K);
    for k = 1:K
        Ct1 = squeeze(logu.C_hist(k,:,1));
        D0(k) = local_dist(Ct1, C_final(k,:), distName);
        if D0(k) == 0, D0(k) = 1; end
    end

    for t = 1:T
        for k = 1:K
            Ctk   = squeeze(logu.C_hist(k,:,t));
            d_abs = local_dist(Ctk, C_final(k,:), distName);
            Dabs(t,k) = d_abs;
            Drel(t,k) = d_abs / D0(k);
        end
    end
end

function d = local_dist(a, b, distName)
    switch lower(distName)
        case 'cityblock'
            d = sum(abs(a - b));
        case 'sqeuclidean'
            d = sum((a - b).^2);
        case 'euclidean'
            d = sqrt(sum((a - b).^2));
        otherwise
            d = sum(abs(a - b));
    end
end
