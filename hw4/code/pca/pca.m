function [eigvector, eigvalue] = PCA(data)
%PCA	Principal Component Analysis
%
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem.
%

% % YOUR CODE HERE
k=size(data,2);
% m = mean(data);
% [N,~]=size(data);
% m_mat = repmat(m,N,1);
% S = (data-m_mat)'*(data-m_mat)./N;
% [V,D]=eig(S);
[u,s,v]=mySVD(data,k);
V=v;
eigvalue = diag(s'*s);
[~,sorted_index] = sort(eigvalue,'descend');
eigvector = V(:,sorted_index);
end