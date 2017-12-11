function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE
[P,N] = size(X);
X=[ones(1,N);X];
[P,N] = size(X);
A = transpose(zeros(size(X)));
for i = 1:N
    A(i,:)=X(:,i)*-y(i);
end
b = -ones(N,1);
w = quadprog(eye(P),[],A,b);
num = sum(abs(X'*w-1)<1);
end
