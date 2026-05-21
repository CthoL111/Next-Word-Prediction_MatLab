words = {"love", "cats"};
a = containers.Map();
a(words{1}) = 5;    % a("love") = 5
a(words{2}) = 3;    % a("cats") = 3

k = keys(a);              % get all keys
v = cell2mat(values(a));  % get all values as array

result = k(v == 3);       % filter keys where value=3
disp(result{1})           % show the key