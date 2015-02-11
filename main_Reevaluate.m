clc
clear all
X_tr = load('q_mark_is_each_collumn_mean.dat');
ldata = load('training_label.dat');
fprintf('Data loading complete!!\n');
n = size(X_tr,1);
l_dim = size(ldata, 2);
folds = 5;

E_cv_set = [];
H_cv_set = [];
F1_cv_set = [];
savedir = './result_re';
loaddir = './result';
for i_y=1:l_dim,
    fprintf('the %d-th label start!!!!!!!!!!\n', i_y);
    site = sprintf('%s/ind_%d.mat',loaddir,i_y);
    L = load(site);
    cv_ind = L.cv_ind;
    y_tr = ldata(:,i_y);
    y_tr = 2.*y_tr - 1;
    fprintf('Get cross-validation index!!!\n');
    site = sprintf('%s/H_cv_set_%d.dat',loaddir,i_y);
    H_set = load(site);
    E_v = [];
    F1_v = [];
    ind_set = [];
    for cv = 1:folds,
        tic
        X_tr_v = X_tr(cv_ind(cv).c,:);
        y_tr_v = y_tr(cv_ind(cv).c);
        X_te_v = X_tr;
        y_te_v = y_tr;
        X_te_v(cv_ind(cv).c,:) = [];
        y_te_v(cv_ind(cv).c) = [];
        H = H_transfer(H_set, cv);
        [label E_temp] = Adaboost_test(X_te_v, y_te_v, H);
        E_v = [E_v E_temp];
        f1_Y = size(find(y_te==1),1);
        f1_P = size(find(label==1),1);
        den = label + y_te_v;
        f1_den = size(find(den==2),1);
        F1_temp = 2*f1_den/(f1_Y+f1_P);
        F1_v = [F1_v F1_temp];
        t = toc;
        fprintf('After fold %d Time goes %2.2f\n sec', cv, t);
    end
    E_cv = sum(E_v)/folds;
    fprintf('At %d-th label E_cv is %2.4f\n', i_y, E_cv);
    F1_cv = sum(F1_v)/folds;
    fprintf('At %d-th label F1_cv is %2.4f\n', i_y, F1_cv);
    E_cv_set = [E_cv_set E_cv];
    F1_cv_set = [F1_cv_set F1_cv];
    save ./re_result/E_cv_set.dat E_cv_set -ascii
    save ./re_result/F1_cv_set.dat F1_cv_set -ascii    
    fprintf('the %d-th label complete!!!!!!!!!!\n', i_y);
end
