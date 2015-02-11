function [ label Eout ] = Adaboost_test( X_te, y_te, H )
%ADABOOST_TEST Summary of this function goes here
%   Detailed explanation goes here
    Nte = size(X_te,1);
    T = size(H,1);
    cc = 0;
    label = [];
    for i=1:Nte,
        H_tr = 0;
        y = y_te(i);

        for t=1:T,
            th = H(t,1);
            d = H(t,2);
            s = H(t,3);
            alpha = H(t,4);
            x = X_te(i,d);
            h = s*sign(x - th);
            H_tr = H_tr + h*alpha;
        end
        if y~=sign(H_tr), cc = cc +1; end
        label = [label;h];
    end
    fprintf('Eout is %2.8f\n', cc/Nte);
    Eout = cc/Nte;
end

