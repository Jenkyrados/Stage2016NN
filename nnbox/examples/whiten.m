function [Xret, Xmu, Var] = whiten(X,fudgefactor,~)
    Xmu = bsxfun(@minus, double(X), mean(X));
    A = Xmu*Xmu'/size(X,2);
    [U,S,V] = svd(A);
    Var = U*diag(1./(diag(S)+fudgefactor).^(1/2))*U';
    Xret = Var*Xmu;
    Var = diag(Var.^2);
end
