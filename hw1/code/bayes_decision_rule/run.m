% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%%load data
load('data');
all_x = cat(2, x1_train, x1_test, x2_train, x2_test);
range = [min(all_x), max(all_x)];
train_x = get_x_distribution(x1_train, x2_train, range);
test_x = get_x_distribution(x1_test, x2_test, range);
[c1,len1] = size(x1_test);
[c2,len2] = size(x2_test);
%% Part1 likelihood: 
l = likelihood(train_x);

% bar(range(1):range(2), l');
% xlabel('x');
% ylabel('P(x|\omega)');
% axis([range(1) - 1, range(2) + 1, 0, 0.5]);

%TODO
%compute the number of all the misclassified x using maximum likelihood decision rule

error = 0;
for i =1:len1
    sample = x1_test(i)-range(1)+1;
    if l(1,sample) < l(2,sample)
        error= error+1;
    end
end
for i = 1:len2
    sample = x2_test(i)-range(1)+1;
    if l(1,sample) > l(2,sample)
        error = error+1;
    end
end
fprintf('likelihood:\n')
fprintf('error:%d\n',error)

%% Part2 posterior:
p = posterior(train_x);

% bar(range(1):range(2), p');
% xlabel('x');
% ylabel('P(\omega|x)');
% axis([range(1) - 1, range(2) + 1, 0, 1.2]);

%TODO
%compute the number of all the misclassified x using optimal bayes decision rule
error=0;
for i =1:len1
    sample = x1_test(i)-range(1)+1;
    if p(1,sample) < p(2,sample)
        error= error+1;
    end
end
for i = 1:len2
    sample = x2_test(i)-range(1)+1;
    if p(1,sample) > p(2,sample)
        error = error+1;
    end
end
fprintf('poseterior:\n')
fprintf('error:%d\n',error)
x_test = cat(2,x1_test,x2_test);
%% Part3 risk:
risk = [0, 1; 2, 0];
fprintf('mininal risk is %f\n',min(sum(risk(1,:)*p(:,x_test-range(1)+1)),sum(risk(2,:)*p(:,x_test-range(1)+1))))
%TODO
%get the minimal risk using optimal bayes decision rule and risk weights
