function [cl, yhat] = kmeans_predict_consistent(model, X)
    % Standardise et pondère comme à l'entraînement, puis affecte par distance
    Xz = (X - model.mu)./model.sg;
    if ~isempty(model.w)
        Xz = Xz .* model.w;
    end
    D = pdist2_compat(Xz, model.C, model.dist);
    [~, cl] = min(D, [], 2);
    if isfield(model,'map') && ~isempty(model.map)
        yhat = model.map(cl);
    else
        yhat = cl; % au besoin
    end
end