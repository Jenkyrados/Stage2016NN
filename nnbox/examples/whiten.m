function [Xret] = whiten(X,fudgefactor,~)
    Xt = bsxfun(@minus, double(X), mean(X));
    A = Xt*Xt'/size(X,2);
    [U,S,V] = svd(A);
    Xret = U*diag(1./(diag(S)+fudgefactor).^(1/2))*U'*Xt;
end
