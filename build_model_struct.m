function model = build_model_struct(out, Xref, w, distName)
    % Reconstruit la même standardisation que dans kmeans_improve
    mu = mean(Xref,1);
    sg = std(Xref,[],1); sg(sg==0)=1;
    model = struct();
    model.C    = out.C;           % centroïdes dans l'espace standardisé + pondéré
    model.map  = out.map;         % mapping cluster->label
    model.mu   = mu;              % standardisation (pour prédire)
    model.sg   = sg;
    model.w    = w(:).';          % 1 x d ou vide
    model.dist = distName;        % 'cityblock' ou 'sqeuclidean'
end