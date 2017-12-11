function w = logistic_r(X, y, lambda)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
[P,N] = size(X);
X = [ones(1,N);X];
[P,N] = size(X);
eps  = 0.1;
w = transpose(rand(1,P));
T =zeros(N,P);
for i = 1:N
    T(i,:) = (y(i)-(1/(1+exp(-w'*X(:,i)))))*X(:,i);
end
E = sum(T);
E = E + 2*lambda*w';
iter = 0;
while(any(find(abs(E)>eps)) && iter < 100)
    w=w+0.01*E';
    T =zeros(N,P);
    for i = 1:N
        T(i,:) = (y(i)-(1/(1+exp(-w'*X(:,i)))))*X(:,i);
    end
    E = sum(T);
    E = E + 2*lambda*w';
    iter = iter+1;
end

end
