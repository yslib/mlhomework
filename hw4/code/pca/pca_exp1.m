load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');

% YOUR CODE HERE

% 1. Feature preprocessing
% 2. Run PCA
% 3. Visualize eigenface
% 4. Project data on to low dimensional space
% 5. Run KNN in low dimensional space
% 6. Recover face images form low dimensional space, visualize them
fea_Train = zscore(fea_Train);
fea_Test = zscore(fea_Test);
[v_train,~]=pca(fea_Train);
v_train=real(v_train);
show_face(v_train');
[v_test,~] = pca(fea_Test);
v_test=real(v_test);
%show_face(v_test');
d = [8,16,32,64,128];
for i = 1:size(d,2)
    k = d(i);
    low_dim_train = v_train(:,1:k)'*fea_Train';
    
    low_dim_test = v_train(:,1:k)'*fea_Test';
    y= knn(low_dim_test,low_dim_train,gnd_Train',1);
    test_num = size(low_dim_test,2);
    err_num = sum(y ~= gnd_Test');
    fprintf("when d = %d, test error = %f%%\n",k,(err_num/test_num)*100);
end
    
    
