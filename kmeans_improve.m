function out = kmeans_improve(X, K, varargin)
% KMEANS_IMPROVE  K-means amélioré avec k-means++, standardisation, pondération,
%                 multiples réplications, gestion des clusters vides et arrêt par tolérance.
%
% out = kmeans_improve(X, K, 'Name',Value, ...)
%   X : N x d (N échantillons, d features)
%   K : nombre de clusters
%
% Options (Name-Value) :
%   'Distance'    : 'sqeuclidean' (defaut) ,'euclidean'ou 'cityblock'
%   'Replicates'  : entier >=1 (defaut 10)
%   'MaxIter'     : entier (defaut 300)
%   'Tol'         : tolérance déplacement centroïdes (defaut 1e-4)
%   'Standardize' : true/false (defaut true)
%   'FeatureWeights' : 1 x d (pondération des features, defaut [])
%   'Labels'      : N x 1 (labels réels pour mapper les clusters et calculer l’accuracy)
%
% Sorties (struct out) :
%   out.idx       : N x 1 indices de cluster (1..K)
%   out.C         : K x d centroïdes
%   out.inertia   : somme des distances au centroïde (objective)
%   out.repBest   : réplication gagnante
%   out.acc       : accuracy (%) si Labels fournis, sinon []
%   out.map       : mapping cluster->label si Labels fournis, sinon []
%
% Auteur : vous :)

% --- Parse options
p = inputParser;
addParameter(p,'Distance','sqeuclidean');
addParameter(p,'Replicates',10);
addParameter(p,'MaxIter',300);
addParameter(p,'Tol',1e-4);
addParameter(p,'Standardize',true);
addParameter(p,'FeatureWeights',[]);
addParameter(p,'Labels',[]);
parse(p,varargin{:});
distName = validatestring(p.Results.Distance, {'sqeuclidean','cityblock','euclidean'});
R        = p.Results.Replicates;
MaxIter  = p.Results.MaxIter;
Tol      = p.Results.Tol;
doStd    = p.Results.Standardize;
w        = p.Results.FeatureWeights;
y        = p.Results.Labels;

% --- Prétraitement
X = double(X);
[N,d] = size(X);
if doStd
    mu = mean(X,1);
    sg = std(X,[],1); sg(sg==0)=1;
    Xz = (X - mu)./sg;
else
    Xz = X;
end
if ~isempty(w)
    w = w(:).';
    if numel(w) ~= d, error('FeatureWeights must be 1 x d'); end
    if any(w<=0), error('FeatureWeights must be > 0'); end
    % Pondération = équivalent à scaler les colonnes
    Xz = Xz .* w;
end

best.inertia = inf;
best.idx = []; best.C = []; best.repBest = 1;

for rep = 1:R
    % --- Initialisation k-means++
    C = kpp_init(Xz, K, distName);

    % --- Itérations de Lloyd
    prevC = C;
    for it = 1:MaxIter
        % Assignation
        D = pdist2_compat(Xz, C, distName);
        [~, idx] = min(D, [], 2);

        % Mise à jour des centroïdes
        Cnew = zeros(K,d);
        for k = 1:K
            mk = (idx==k);
            if any(mk)
                Cnew(k,:) = mean(Xz(mk,:), 1);
            else
                % cluster vide : ré-ensemencer sur le point le plus éloigné
                [~, far] = max(min(D,[],2));
                Cnew(k,:) = Xz(far,:);
            end
        end

        % Critère d’arrêt
        shift = max(sqrt(sum((Cnew - prevC).^2, 2))); % max norme L2 par centroïde
        prevC = Cnew;
        C = Cnew;
        if shift < Tol, break; end
    end

    % Inertie (somme distances au centroïde)
    D = pdist2_compat(Xz, C, distName);
    inertia = sum(D(sub2ind(size(D), (1:N)', idx)));

    if inertia < best.inertia
        best.inertia = inertia;
        best.idx = idx;
        best.C = C;
        best.repBest = rep;
    end
end

out.idx     = best.idx;
out.C       = best.C;
out.inertia = best.inertia;
out.repBest = best.repBest;

% --- Évaluation optionnelle si labels fournis
out.acc = [];
out.map = [];
if ~isempty(y)
    y = y(:);
    if numel(y) ~= N
        warning('Labels ignored: size mismatch (N=%d, numel(y)=%d).', N, numel(y));
    else
        % Mapping cluster->label par vote majoritaire
        map = zeros(K,1);
        for k = 1:K
            mk = (best.idx==k);
            if any(mk)
                map(k) = mode(y(mk));
            else
                map(k) = mode(y);
            end
        end
        yhat = map(best.idx);
        out.acc = 100*mean(yhat == y);
        out.map = map;
    end
end
end

% --------- k-means++ init locale ---------
function C = kpp_init(X, K, distName)
[N,~] = size(X);
C = zeros(K, size(X,2));
% 1er centroïde au hasard
i1 = randi(N);
C(1,:) = X(i1,:);
% suivants : probabilité ∝ D^2 (ou D pour cityblock, on garde D^2 par cohérence)
D = pdist2_compat(X, C(1,:), distName).^2;
for k = 2:K
    p = D ./ sum(D);
    % tirage pondéré
    edges = cumsum(p);
    r = rand;
    i = find(edges >= r, 1, 'first');
    C(k,:) = X(i,:);
    % mettre à jour les distances au plus proche centroïde
    D = min(D, pdist2_compat(X, C(k,:), distName).^2);
end
end
