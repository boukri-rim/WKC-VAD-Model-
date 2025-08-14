function D = pdist2_compat(A, B, distName)
% Compatibilité totale (évite l'erreur 'sqeuclidean' inconnu)
switch distName
    case 'sqeuclidean'
        % L2^2 via identité (a-b)^2 = a^2 + b^2 - 2ab
        AA = sum(A.^2,2);
        BB = sum(B.^2,2).';
        D = max(0, bsxfun(@plus, AA, BB) - 2*(A*B.'));  % >=0
    case 'euclidean'
        % L2 (non carrée)
        AA = sum(A.^2,2);
        BB = sum(B.^2,2).';
        D2 = max(0, bsxfun(@plus, AA, BB) - 2*(A*B.'));
        D = sqrt(D2);
    case 'cityblock'
        % L1 (mémoire O(NK), ok pour taille modérée)
        nA = size(A,1); nB = size(B,1);
        D = zeros(nA, nB);
        for j = 1:size(A,2)
            D = D + abs(A(:,j) - B(:,j).');
        end
    otherwise
        error('Unsupported distance "%s".', distName);
end
end