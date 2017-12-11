load('TDT2_data', 'fea', 'gnd');

% YOUR CODE HERE
options = [];
options.NeighborMode = 'KNN';
options.k = 40;
options.WeightMode = 'HeatKernel';
W = constructW(fea,options);
label = unique(gnd);
idx=gnd;
for i = 1:length(label)
    idx(logical(idx==label(i)))=i;
end
sum_nmi1 =0;
sum_nmi2=0;
c=10;
for i = 1:c
    idx1 = spectral(W, 6);
    idx2 = litekmeans(fea, 6);
    sum_nmi1=sum_nmi1+MutualInfo(idx,idx1);
    sum_nmi2 =sum_nmi2+ MutualInfo(idx,idx2);
end
fprintf("NMI between GND and Spectral Clustering is %f\n",sum_nmi1/c);
fprintf("NMI between GND and Kmeans Clustering is %f\n",sum_nmi2/c);