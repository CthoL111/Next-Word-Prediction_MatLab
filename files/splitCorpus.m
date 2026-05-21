% PART 4: Split tokens into train and test sets
function [trainTokens, testTokens] = splitCorpus(tokens, trainRatio)
    if nargin < 2, trainRatio = 0.8; end
    splitIdx    = floor(numel(tokens) * trainRatio);
    trainTokens = tokens(1:splitIdx);
    testTokens  = tokens(splitIdx+1:end);
    fprintf('Train: %d tokens | Test: %d tokens\n', numel(trainTokens), numel(testTokens));
end
