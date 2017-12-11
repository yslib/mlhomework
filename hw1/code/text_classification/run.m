%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
%N is the size of vocabulary.
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;
%Do smoothing
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;

%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier
total = num_ham_train+num_spam_train;
prior=[num_ham_train/total,num_spam_train/total];

%convert sparse matrix to full matrix
test_ham_full = full(ham_test);
test_spam_full= full(spam_test);

c_ham = size(test_ham_full,1);
c_spam = size(test_spam_full,1);

l = likelihood(x);
[C,N]=size(x);
p_ham = zeros(c_ham,C);
p_spam = zeros(c_spam,C);

%Naive Bayes
%ham posterior
for i = 1:c_ham         %for c_ham test email
    for j = 1:C %for C classes
        for k = find(test_ham_full(i,:))
            count = test_ham_full(i,k);
            p_ham(i,j) = p_ham(i,j)+count*log(l(j,k)); %
        end
        p_ham(i,j)=p_ham(i,j)+log(prior(j));
    end
end

%spam posterior
for i = 1:c_spam %for c_spam test email
    for j = 1:C % for C classes
        for k = find(test_spam_full(i,:))
            count = test_spam_full(i,k);
            p_spam(i,j) = p_spam(i,j)+count*log(l(j,k)); %
        end
        p_spam(i,j)=p_spam(i,j)+log(prior(j));
    end
end

%Compute TP FP FN TN
TP=0; FP=0; FN=0; TN=0;
for i = 1:c_ham  %for very test case in ham
    [m,id] = max(p_ham(i,:));   %selecting class with maximum posterior
    if id == 1
        TN=TN+1;
    else
        FN= FN+1;
    end
end
for i = 1:c_spam % for evert test case in spam
    [m,id] = max(p_spam(i,:)); %selecting class with maximum posterior
    if id == 1
        FP=FP+1;
    else
        TP= TP+1;
    end
end

%Compute P(wordi|SPAM)/P(wordi/HAM)
res = l(1,:)./l(2,:);
[top_10,top_10_id]=sort(res,'descend');
top_10_id = top_10_id(1:10);
[id2word,ids]=textread('all_word_map.txt','%s %d','headerlines',2);
fprintf('top 10 words with highest P(wordi|SPAM)/P(wordi/HAM):\n');
id2word(top_10_id)

fprintf('TP:%d\nFP:%d\nFN:%d\nTN:%d\n',TP,FP,FN,TN);
%precison and recall
fprintf('precision:%f\nrecall:%f\n',TP/(TP+FP),TP/(TP+FN));