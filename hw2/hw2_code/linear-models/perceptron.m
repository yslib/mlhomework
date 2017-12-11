function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
[P,N]=size(X);
X = [ones(1,N);X];
[P,N] = size(X);
w = rand(P,1);
Im = find(transpose(w)*X.*y<0);
iter = 0;
while(~isempty(Im))
    w = w+X(:,Im(1))*y(Im(1));
    iter=iter+1;
    Im = find(transpose(w)*X.*y<0);
end

end
