function [ H ] = Adaboost_train( X_tr, y_tr, T )
%ADABOOST_TRAIN Summary of this function goes here
%   Detailed explanation goes here

Ntr = size(X_tr,1);
u = ones(Ntr,1)./Ntr;
H = zeros(T,6);
eps_set = [];

for t=1:T,
    fprintf('now is computing t = %d\n', t);
    [th Acc d s] = Decision_stump(X_tr, y_tr, u);
    count = 0;
    c = 0;
    
    for i=1:Ntr,
        x = X_tr(i,d);
        y = y_tr(i);
        h = s*sign(x - th);
        u_v = u(i);
        if h~=y, 
            count = count + u_v;
            c = c + 1;
        end
    end
    eps = count/sum(u);
    H(t,6) = sum(u);
    eps_set = [eps_set;eps];
    alpha = 0.5*log((1 - eps)/eps);

    for i=1:Ntr,
        x = X_tr(i,d);
        y = y_tr(i);
        h = s*sign(x - th);
        u(i) = u(i)*exp(-alpha*y*h);
    end
    H(t,1) = th;
    H(t,2) = d;
    H(t,3) = s;
    H(t,4) = alpha;
    H(t,5) = c/Ntr;
end

end

