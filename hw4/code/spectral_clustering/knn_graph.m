function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE
[n,p]=size(X);
D = pdist2(X,X);
D(logical(eye(n)))=inf;
W=zeros(n,n);
for i=1:n
    t = D(i,:);
    [~,I]=sort(t);
    I=I(1:k);
    W(i,I(t(I)<threshold))=1;
end
end
