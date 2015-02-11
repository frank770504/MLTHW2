function [ beta Y_tr_pre Ein] = RridgeRegClassfy( X_tr,  Y_tr, gamma, lambda)
%RRIDGEREGCLASSFY Summary of this function goes here
%   Detailed explanation goes here
    N_tr = size(X_tr,1);
    
    K_mat = zeros(N_tr,N_tr);

    for n=1:N_tr,
        for m=1:N_tr,
            xn = X_tr(n,:)';
            xm = X_tr(m,:)';
            K_mat(n,m) = KernelRBF(xn, xm, gamma);
        end
    end

    %beta = inv(eye(N_tr)*lambda + K_mat)*Y_tr;
    beta = (eye(N_tr)*lambda + K_mat)\Y_tr;
    
    Y_tr_pre = zeros(N_tr,1);
    for i=1:N_tr,
        xi = X_tr(i,:)';
        y = 0;
        for n=1:N_tr,
            xn = X_tr(n,:)';
            y = y + beta(n)*KernelRBF(xi, xn, gamma);
        end
        Y_tr_pre(i) = sign(y);
    end
    
    %Beta = repmat(beta, 1, size(X_tr, 2));
    %w = sum(X_tr.*Beta)';

    %Y_tr_pre = sign(X_tr*w);
    Ein_l = (Y_tr ~= Y_tr_pre);
    Ein = sum(Ein_l)/N_tr;

end

