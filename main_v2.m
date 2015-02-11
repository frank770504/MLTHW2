clc
%clear all
rdata = load('hw2_adaboost_train.dat');
X_tr = rdata(:,1:2);
y_tr = rdata(:,end);
tic
H = Adaboost_train(X_tr, y_tr, 300);
toc
rdata = load('hw2_adaboost_test.dat');
X_te = rdata(:,1:2);
y_te = rdata(:,end);
Eout = [];
Ein = [];
for i=1:300,
    [label eout] = Adaboost_test(X_te, y_te, H(1:i,:));
    [label ein] = Adaboost_test(X_tr, y_tr, H(1:i,:));
    Eout = [Eout;eout];
    Ein = [Ein;ein];
end

plot(H(:,5))
hold on
plot(H(:,6),'r')
plot(Eout,'k')
plot(Ein,'c')
legend('Ein_t','U_t','Eout','Ein');
% f1_Y = size(find(y_te==1),1);
% f1_P = size(find(label==1),1);
% den = label + y_te;
% f1_den = size(find(den==2),1);
% F1_temp = 2*f1_den/(f1_Y+f1_P);