function w = prepare_feature_weights(F, d)
    % Essaie de convertir F en 1 x d valide; sinon renvoie []
    w = [];
    if ~exist('F','var') || isempty(F), return; end
    try
        cand = double(F(:).');
        if numel(cand) == d
            w = cand; return;
        end
        if size(F,1) >= 1 && size(F,2) >= d
            w = double(F(1,1:d)); return;
        end
        if size(F,1) >= d && size(F,2) >= 1
            w = double(F(1:d,1)).'; return;
        end
    catch
        w = [];
    end
    if ~isempty(w)
        w(w<=0) = 1; % évite poids nuls/négatifs
    end
end