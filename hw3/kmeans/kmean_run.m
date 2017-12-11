clear;
load('./kmeans/kmeans_data.mat');
min_sd = 999999999;
max_sd = -1;
min_idx =[];min_ctrs=[];min_iter_ctrs=[];
max_idx=[];max_ctrs=[];max_iter_ctrs=[];
for i = 1:1000
    [idx,ctrs,iter_ctrs]=kmeans(X,2);
    cur_sd = 0;
    for j = 1:2
        cur_sd =cur_sd+ sum(sqrt(sum((X(idx==j,:)-repmat(ctrs(j,:),sum(idx==j),1)).^2,2)));
    end
    if(cur_sd > max_sd)
        max_sd = cur_sd;
        max_idx = idx;
        max_ctrs = ctrs;
        max_iter_ctrs = iter_ctrs;
    end
    if(cur_sd<min_sd)
        min_sd = cur_sd;
        min_idx=idx;
        min_ctrs=ctrs;
        min_iter_ctrs=iter_ctrs;
    end
end
%kmeans_plot(X,min_idx,min_ctrs,min_iter_ctrs);
kmeans_plot(X,max_idx,max_ctrs,max_iter_ctrs);