function [Xtt] = tucker2( X, ranks )

    Xtt = cell(length(ranks)-1,1);
    dim = size(X);
    N = ndims(X);
    
    % compute matrices of leaves
    for n = 1:2
        W = reshape(permute(double(X),[n [1:n-1,n+1:N]]),dim(n), prod(dim)/dim(n));
        [U,S,V] = svd(W, 'econ');
        V=V*S;
        Xtt{n} = U(:,1:ranks(n));
        SV = V(:,1:ranks(n))';
        X = reshape(SV, [ranks(n:-1:1) dim(n+1:N)]);
        dim(n) = ranks(n);
    end
    
    Xtt{3} = permute(X,[2 1 3 4]);
    
end
    