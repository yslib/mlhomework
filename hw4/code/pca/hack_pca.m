function img = hack_pca(filename)
% Input: filename -- input image file name/path
% Output: img -- image without rotation

img_r = double(imread(filename));

% YOUR CODE HERE
imgg = mat2gray(img_r);
% imshow(imgg);
[h,w]=size(imgg);
[a,b]=find(imgg<0.7);
img_coord = [b,h+1-a];
ctred_img_coord = img_coord-repmat(mean(img_coord),size(img_coord,1),1);
[v,d] = pca(ctred_img_coord)
v=-v;
angle = atan2(v(2,1),v(1,1))*180/pi
img = imrotate(imgg,-angle);
end