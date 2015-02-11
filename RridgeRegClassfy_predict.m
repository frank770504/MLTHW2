function [ Y_te_pre E_out ] = RridgeRegClassfy_predict( X_te, Y_te, beta, gamma, X_tr)
%RRIDGEREGCLASSFY_PREDICT Summary of this function goes here
%   Detailed explanation goes here
    N_te = size(X_te,1);
    N_tr = size(beta,1);
    Y_te_pre = zeros(N_te,1);
    for i=1:N_te,
        xi = X_te(i,:)';
        y = 0;
        for n=1:N_tr,
            xn = X_tr(n,:)';
            y = y + beta(n)*KernelRBF(xi, xn, gamma);
        end
        Y_te_pre(i) = sign(y);
    end
    Eout_l = (Y_te ~= Y_te_pre);
    E_out = sum(Eout_l)/N_te;

end

