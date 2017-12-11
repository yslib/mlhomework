function y = knn(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.
%

% YOUR CODE HERE
[~,N]=size(y_train);
distmat=transpose(pdist2(X',X_train')); % N_train by N_test
[~,sorted_index]=sort(distmat,'ascend');
sorted_index_top_k = sorted_index(1:K,:);
if K == 1
    y = y_train(sorted_index_top_k);
    return;
end
[~,label_index] = max(hist(sorted_index_top_k,N),[],1);
y=y_train(label_index);
end


