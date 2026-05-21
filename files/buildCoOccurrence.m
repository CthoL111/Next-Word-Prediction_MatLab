% PART 3: Build Co-occurrence Matrix for Word Vectors
% Words that appear near each other get higher scores
function coMatrix = buildCoOccurrence(tokens, vocab, windowSize)
    if nargin < 3, windowSize = 2; end
    n         = numel(vocab);
    wordIndex = containers.Map(vocab, 1:n);
    coMatrix  = zeros(n, n);

    for i = 1:numel(tokens)
        if ~isKey(wordIndex, tokens{i}), continue; end
        center = wordIndex(tokens{i});

        % Look at surrounding words within window
        for j = max(1,i-windowSize):min(numel(tokens),i+windowSize)
            if j == i, continue; end
            if ~isKey(wordIndex, tokens{j}), continue; end
            context = wordIndex(tokens{j});
            coMatrix(center, context) = coMatrix(center, context) + 1;
        end
    end
end
