function [idx, ctrs, iter_ctrs] = kmeans(X, K)
%KMEANS K-Means clustering algorithm
%
%   Input: X - data point features, n-by-p maxtirx.
%          K - the number of clusters
%
%   OUTPUT: idx  - cluster label
%           ctrs - cluster centers, K-by-p matrix.
%           iter_ctrs - cluster centers of each iteration, K-by-p-by-iter
%                       3D matrix.

% YOUR CODE HERE
[n,p]=size(X);
rdp = randperm(n);  %????
ctrs = X(rdp(1:K),:);   %????k??? k by p
eps = 0.1;
new_ctrs = zeros(size(ctrs));
iter = 1;
while (1)
    iter_ctrs(:,:,iter) = ctrs;
    distmat = pdist2(X,ctrs); % n by k
    [~,idx] = min(distmat,[],2);  % idx n by 1: value :1 to k
    for i=1:K
        new_ctrs(i,:) = sum(X(idx ==i,:))./sum((idx==i));
    end
    if (any(any(abs(ctrs-new_ctrs) < eps))||(iter >3000))
        break;
    end
    iter=iter+1;
    ctrs = new_ctrs;
end

end