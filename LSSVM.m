clc
clear all
rdata = load('hw2_lssvm_all.dat');
N_tr = 400;
Tr = rdata(1:N_tr,:);
X_tr = Tr(:,1:10);
Y_tr = Tr(:,end);

Te = rdata(N_tr+1:end,:);
X_te = Te(:,1:10);
Y_te = Te(:,end);

gamma = [32 2 0.125];
lambda = [0.001 1 1000];

E_col = [];

for ga = gamma,
    for la = lambda,
        [beta Y_tr_pre Ein] = RridgeRegClassfy(X_tr, Y_tr, ga, la);
        [ Y_te_pre E_out ] = RridgeRegClassfy_predict(X_te, Y_te, beta, ga, X_tr);
        E_col = [E_col; ga la Ein E_out];
    end
end