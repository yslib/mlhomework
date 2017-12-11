img = imread('sample2.jpg');
fea = double(reshape(img, size(img, 1)*size(img, 2), 3));
% YOUR (TWO LINE) CODE HERE
k=64;
[idx,ctrs,iter_ctrs]=kmeans(fea,k);
for i = 1:k
    fea(idx==i,:)=repmat(ctrs(i,:),sum(idx==i),1);
end
imshow(uint8(reshape(fea, size(img))));

%Compression Ratio
ctr_hist = hist(idx);
[~,avg]= huffmandict([1:k],hist(idx,k)./size(idx,1));
pix = size(idx,1); 
size1 = pix*avg/8; %encoded with huffman
size2 = k*3; % cluster center
huf = 0;
for i = 1:k
    a = dict{i,2};
    huf = huf+size(a,2)+8; % 8bit for index to cluster center
end
size3 = huf/8; % huffman code
cr = (size1+size2+size3)/(pix*3);
fprintf('compression ratio:%f%%\n',cr*100);

