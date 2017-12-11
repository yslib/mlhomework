function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
[P,N] = size(X);
X = [ones(1,N);X];
[P,N] = size(X);
eps  = 1;
w = transpose(rand(1,P));
T =zeros(N,P);
for i = 1:N
    T(i,:) = (y(i)-(1/(1+exp(-w'*X(:,i)))))*X(:,i);
end
E = sum(T);
iter = 0;
while(any(find(abs(E)>eps)) && iter < 300)
    w=w+E';
    T =zeros(N,P);
    for i = 1:N
        T(i,:) = (y(i)-(1/(1+exp(-w'*X(:,i)))))*X(:,i);
    end
    E = sum(T);
    iter = iter+1;
end

end
