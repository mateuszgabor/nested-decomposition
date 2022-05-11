function [Xtt] = conv_tucker2(X, R1, R2)
    J = [R1 R2];
    [Xtt] = tucker2(X, J);
    Xs = permute(tensor_contraction(tensor_contraction(Xtt{3},Xtt{1},1,2),Xtt{2},1,2),[3 4 1 2]);
    diff = X - Xs;
    r1 = norm(diff(:))/norm(X(:))
end