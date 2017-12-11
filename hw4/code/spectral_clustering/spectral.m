function idx = spectral(W, k)
%SPECTRUAL spectral clustering
%   Input:
%     W: Adjacency matrix, N-by-N matrix
%     k: number of clusters
%   Output:
%     idx: data point cluster labels, n-by-1 vector.

% YOUR CODE HERE
[N1,N2]=size(W);
D = zeros(N1,N2);
D(logical(eye(N1))) = sum(W,1);
L = D-W;
[v,~] = eigs(L,D,k,'SM');
idx = kmeans(v,k);
end
