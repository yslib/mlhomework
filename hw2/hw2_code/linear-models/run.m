% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.
%warning('off');
%% Part1: Preceptron
nRep = 1000; % number of replicates
nTrain = 10; % number of training data
nTest = 1000;
total_iter = 0;
E_test = 0;
E_train = 0;
for i = 1:nRep
    [X_all, y_all, w_f] = mkdata(nTrain+nTest);
    X = X_all(:,1:nTrain);y=y_all(:,1:nTrain);   %train data
    X_test = X_all(:,nTrain+1:nTrain+nTest);y_test = y_all(:,nTrain+1:nTrain+nTest); % test data
    
    [w_g, iter] = perceptron(X, y);
    % Compute training, testing error
    [P,N] = size(X);
    E_train = E_train + sum(transpose(w_g)*[ones(1,N);X].*y<0);
    %[X_test, y_test, w_f_test] = mkdata(nTest);
    [P_test,N_test] = size(X_test);
    E_test = E_test + sum(transpose(w_g)*[ones(1,N_test);X_test].*y_test<0);
    total_iter = total_iter + iter; % Sum up number of iterations
end
avgIter = total_iter/nRep;
fprintf('In Preceptron,result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(X_test, y_test, w_f, w_g, 'Pecertron');
%plotdata(X,y,w_f,w_g,'Percertron');

%% Part2: Preceptron: Non-linearly separable case
% nTrain = 100; % number of training data
% [X, y, w_f] = mkdata(nTrain, 'noisy');
% [w_g, iter] = perceptron(X, y);


%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 100;
E_test = 0;
E_train = 0;
for i = 1:nRep
    [X_all, y_all, w_f] = mkdata(nTrain+nTest);
    X = X_all(:,1:nTrain);y=y_all(:,1:nTrain);   %train data
    X_test = X_all(:,nTrain+1:nTrain+nTest);y_test = y_all(:,nTrain+1:nTrain+nTest); % test data
    
    w_g = linear_regression(X, y);
    % Compute training, testing error
    [P,N] = size(X);
    E_train = E_train + sum(transpose(w_g)*[ones(1,N);X].*y<0);
    [P_test,N_test] = size(X_test);
    E_test = E_test + sum(transpose(w_g)*[ones(1,N_test);X_test].*y_test<0);
end
fprintf('In Linear Regression,result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);
plotdata(X, y, w_f, w_g, 'Linear Regression');
 
%% Part4: Linear Regression: noisy
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 100;
E_test = 0;
E_train = 0;
for i = 1:nRep
    [X_all, y_all, w_f] = mkdata(nTrain+nTest,'noisy');
    X = X_all(:,1:nTrain);y=y_all(:,1:nTrain);   %train data
    X_test = X_all(:,nTrain+1:nTrain+nTest);y_test = y_all(:,nTrain+1:nTrain+nTest); % test data
    w_g = linear_regression(X, y);
    % Compute training, testing error

    [P,N] = size(X);
    E_train = E_train + sum(transpose(w_g)*[ones(1,N);X].*y<0);
    [P_test,N_test] = size(X_test);
    E_test = E_test + sum(transpose(w_g)*[ones(1,N_test);X_test].*y_test<0);
end
fprintf('In Linear Regression(Noisy),result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);
plotdata(X, y, w_f, w_g, 'Linear Regression: noisy');

%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');
nTrain = size(X,2);
nTest = size(X_test,2);
nRep = 1;
E_test = 0;
E_train = 0;
w = linear_regression(X, y);
% Compute training, testing error

[P,N] = size(X);
E_train = E_train + sum(transpose(w)*[ones(1,N);X].*y<0);
[P_test,N_test] = size(X_test);
 E_test = E_test + sum(transpose(w)*[ones(1,N_test);X_test].*y_test<0);

fprintf('In Linear Regression(poly_fit),result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);
plotdata(X, y, w, w, 'Linear Regression: poly_fit train');
plotdata(X_test, y_test, w, w, 'Linear Regression: poly_fit test');
% poly_fit with transform
nRep = 1;
E_test = 0;
E_train = 0;
X_t = [X;X(1,:).*X(2,:);X(1,:).*X(1,:);X(2,:).*X(2,:)];% CHANGE THIS LINE TO DO TRANSFORMATION
X_test_t = [X_test;X_test(1,:).*X_test(2,:);X_test(1,:).*X_test(1,:);X_test(2,:).*X_test(2,:)];% CHANGE THIS LINE TO DO TRANSFORMATION
w = linear_regression(X_t, y);

% Compute training, testing error
[P_t,N_t] = size(X_t);
E_train = E_train + sum(transpose(w)*[ones(1,N_t);X_t].*y < 0);
[P_test_t,N_test_t] = size(X_test_t);
E_test = E_test + sum(transpose(w)*[ones(1,N_test_t);X_test_t].*y_test < 0);
fprintf('In Linear Regression(poly_fit with transform),result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);


%% Part6: Logistic Regression
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 100;
E_test = 0;
E_train = 0;
for i = 1:nRep
    [X_all, y_all, w_f] = mkdata(nTrain+nTest);
    y_all(find(y_all == -1))=0;
    X = X_all(:,1:nTrain);y=y_all(:,1:nTrain);   %train data
    X_test = X_all(:,nTrain+1:nTrain+nTest);y_test = y_all(:,nTrain+1:nTrain+nTest); % test data
    
    w_g = logistic(X, y);
    
    % Compute training, testing error
    [P,N] = size(X);
    E_train = E_train + sum(xor(1./(1+exp(-w_g'*[ones(1,N);X]))>0.5,y));
    [P_test,N_test] = size(X_test);
    E_test = E_test + sum(xor(1./(1+exp(-w_g'*[ones(1,N_test);X_test]))>0.5,y_test));
end
fprintf('In Logistic Regression,result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);
y(find(y == 0))=-1;
plotdata(X, y, w_f, w_g, 'Logistic Regression train');
y_test(y_test==0)=-1;
plotdata(X_test,y_test,w_f,w_g,'Logistic Regression test');
%
 %% Part7: Logistic Regression: noisy
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 100;
E_test = 0;
E_train = 0;
for i = 1:nRep
    [X_all, y_all, w_f] = mkdata(nTrain+nTest,'noisy');
    y_all(find(y_all == -1))=0;
    X = X_all(:,1:nTrain);y=y_all(:,1:nTrain);   %train data
    X_test = X_all(:,nTrain+1:nTrain+nTest);y_test = y_all(:,nTrain+1:nTrain+nTest); % test data
    
    w_g = logistic(X, y);
    
    % Compute training, testing error
    [P,N] = size(X);
    E_train = E_train + sum(xor(1./(1+exp(-w_g'*[ones(1,N);X]))>0.5,y));
    [P_test,N_test] = size(X_test);
    E_test = E_test + sum(xor(1./(1+exp(-w_g'*[ones(1,N_test);X_test]))>0.5,y_test));
end
fprintf('In Logistic Regression,result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);
y(find(y == 0))=-1;
plotdata(X, y, w_f, w_g, 'Logistic Regression noisy train');
y_test(y_test==0)=-1;
plotdata(X_test,y_test,w_f,w_g,'Logistic Regression noisy test');
%% Part8: SVM
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 1000;
E_train = 0;
E_test = 0;
sum_sv = 0;
for i = 1:nRep
    [X_all, y_all, w_f] = mkdata(nTrain+nTest);
    X = X_all(:,1:nTrain);y=y_all(:,1:nTrain);   %train data
    X_test = X_all(:,nTrain+1:nTrain+nTest);y_test = y_all(:,nTrain+1:nTrain+nTest); % test data
    
    [w_g, num_sc] = svm(X, y);
    % Compute training, testing error
    [P,N]=size(X);
    E_train = E_train + sum(w_g'*[ones(1,N);X].*y <0);
    [P_test,N_test] = size(X_test);
    E_test = E_test + sum(w_g'*[ones(1,N_test);X_test].*y_test<0);
    % Sum up number of support vectors
    sum_sv = sum_sv+num_sc; 
end
fprintf('In SVM,result with %d training sample and %d test sample and %d repeations\n',nTrain,nTest,nRep);
fprintf('E_train is %f (%d:%d), E_test is %f(%d:%d).\n',E_train/(nRep*nTrain), E_train,nRep*nTrain,E_test/(nRep*nTest), E_test,nRep*nTest);
fprintf('The average of SVs if %d with %d repeations\n',sum_sv/nRep,nRep);
plotdata(X, y, w_f, w_g, 'SVM train');
plotdata(X_test,y_test,w_f,w_g,'SVM test')
