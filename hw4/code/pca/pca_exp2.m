% YOUR CODE HERE
load('ORL_data', 'fea_Train', 'gnd_Train', 'fea_Test', 'gnd_Test');
[v_train,~]=pca(fea_Train);
v_train=real(v_train);
show_face(fea_Train);
d = [8,16,32,64,128];
X = fea_Train';
for i = 1:size(d,2)
    k = d(i);
    V = v_train(:,1:k);
    low_dim_train = V'*X;
    rec_mat = V*low_dim_train;
    show_face(rec_mat');
    %fprintf("when d = %d, reconstruction error = %f.\n",k,sum(sum((fea_Train-rec_mat').^2))/(size(fea_Train,2)));
end