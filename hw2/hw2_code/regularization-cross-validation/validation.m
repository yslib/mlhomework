%% Ridge Regression
warning('off');
load('digit_train', 'X', 'y');

% Do feature no rmalization

% X = X';
% s = sum(X);
% X(:,find(s==0))=ones(size(X,1),size(find(s==0),2));
% X=bsxfun(@rdivide,X,sum(X));
% X=X';

for i = 1:size(X,1)
    s = std(X(i,:));
    if s ~= 0
        X(i,:)=(X(i,:)-mean(X(i,:)))./s;
    end
end
% ...

% Do LOOCV
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
[P,N]=size(X);
min_E_val = 9999999999;
for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        X_ = X; y_ = y;
        X_(:,j) = [];
        y_(:,j)=[]; % take point j out of X
        w = ridge(X_, y_, lambdas(i));
        E_val = E_val + (y(:,j)-[1;X(:,j)]'*w)^2;
        %E_val = E_val + (y(:,j)*[1;X(:,j)]'*w<0);
    end
    % Update lambda according validation error
    fprintf('In Ridge Regression,when lambda = %f, E_val = %f wTw = %f.\n',lambdas(i),E_val,w'*w);
    if min_E_val > E_val
        min_E_val = E_val;
        lambda = lambdas(i);
    end
end
fprintf('Ridge Regression lambda chosed by LOOCV is %f\n',lambda);
% Compute training error
nRep = 1;
E_train=0;
E_test= 0;
w_g = ridge(X,y,lambda);
[P,N] = size(X);
nTrain = N;
E_train = E_train + sum(transpose(w_g)*[ones(1,N);X].*y<0);
% lambda = 0;
w0 = ridge(X,y,0);
fprintf('when lambda = 0,wTw = %f\n',w0'*w0);


load('digit_test', 'X_test', 'y_test');
% Do feature normalization
% X_test = X_test';
% ss = sum(X_test);
% X_test(:,find(ss==0))=ones(size(X_test,1),size(find(ss==0),2));
% X_test = bsxfun(@rdivide,X_test,sum(X_test));
% X_test=X_test';

for i = 1:size(X_test,1)
    s = std(X_test(i,:));
    if s ~= 0
        X_test(i,:)=(X_test(i,:)-mean(X_test(i,:)))/s;
    end
end
% ...
% Compute test error
[P_test,N_test] = size(X_test);
nTest = N_test;

E_test = E_test + sum(transpose(w_g)*[ones(1,N_test);X_test].*y_test<0);

fprintf('In Ridge Regression,result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);

%show_digit(X_test);
%% Logistic
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
nTrain = size(X,2);
nTest = size(X_test,2);
nRep = 1;
[P,N]=size(X);
min_E_val = 9999999999;
E_val = 0;
y(find(y == -1)) = 0;
E_train =0;
E_test = 0;

for i = 1:length(lambdas)
    E_val = 0;
    for j = 1:size(X, 2)
        y(find(y == -1)) = 0;
        X_ = X; y_ = y;
        X_(:,j) = [];
        y_(:,j)=[]; % take point j out of X
        w_g = logistic_r(X_, y_, lambdas(i));
        %fprintf('%d %d\n',i,j);
        %E_val = E_val + (y(:,j)*[1;X(:,j)]'*w < 0);
        %E_val = E_val+(y(:,i)-w_g'*X(:,j))^2;
        E_val = E_val + sum(xor(1./(1+exp(-w_g'*[1;X(:,j)]))>0.5,y(j)));
    end
     % Update lambda according validation error
     fprintf('In logistic Regression with Reg,when lambda = %f, E_val = %f\n',lambdas(i),E_val);
    if min_E_val > E_val
        min_E_val = E_val;
        lambda = lambdas(i);
    end
end
y(find(y == 0)) = -1;
fprintf('Logistic:lambda chosed by LOOCV is %f\n',lambda);
w_g = logistic_r(X,y,lambda);
[P,N]=size(X);
E_train = E_train + sum(1./(1+exp(-w_g'*[ones(1,N);X]))>0.5~=y);
%E_train = E_train + sum()
[P_test,N_test]=size(X_test);
E_test = E_test + sum(1./(1+exp(-w_g'*[ones(1,N_test);X_test]))>0.5~=y_test);
fprintf('In Logistic Regression with regulartion,result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);


%% SVM with slack variable
