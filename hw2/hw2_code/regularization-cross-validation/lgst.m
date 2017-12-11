function f = lgst(y,x,w)
    f = y*exp(-y*w'*x)/(1+exp(-y*w'*x));
end