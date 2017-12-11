% function p = posterior2(x,prior)
% [C, N] = size(x);
% if C ~= size(prior,2)
%     fprintf('dimension of prior is not equal to the classes of training data');
%     return;
%     
% p=zeros(C,C); % C by C
% l = likelihood(x); %C by N
% for i = 1:C
%     for j = 1:P
%         for k = 1:N
%             p(i,j)=p(i,j)+log(l(i,k));
%         end
%         p(i,j)=p(i,j)+log(prior(j));
%     end
%     p(i,:)=p(i,:)./sum(p(i,:)); % normalized
% end
% end