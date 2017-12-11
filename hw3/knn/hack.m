function digits = hack(img_name)
%HACK Recognize a CAPTCHA image
%   Inputs:
%       img_name: filename of image
%   Outputs:
%       digits: 1x5 matrix, 5 digits in the input CAPTCHA image.

load('./knn/hack_data');
% YOUR CODE HERE
digits = knn(extract_image(img_name),X,y,1);
end