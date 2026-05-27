% ============================================================
% PART 3: Vector Representations (25%)
% ============================================================

% --- One-Hot Encoding ---
function vectors = oneHotEncode(vocab)
    n       = numel(vocab);
    vectors = containers.Map();
    for i = 1:n
        vec      = zeros(1, n);
        vec(i)   = 1;
        vectors(vocab{i}) = vec;
    end
end

% --- Co-occurrence Matrix ---
function coMatrix = buildCoOccurrence(tokens, vocab, windowSize)
    if nargin < 3, windowSize = 2; end
    n         = numel(vocab);
    wordIndex = containers.Map(vocab, 1:n);
    coMatrix  = zeros(n, n);
    for i = 1:numel(tokens)
        if ~isKey(wordIndex, tokens{i}), continue; end
        center = wordIndex(tokens{i});
        for j = max(1,i-windowSize) : min(numel(tokens),i+windowSize)
            if j == i, continue; end
            if ~isKey(wordIndex, tokens{j}), continue; end
            context = wordIndex(tokens{j});
            coMatrix(center, context) = coMatrix(center, context) + 1;
        end
    end
end

% --- Cosine Similarity ---
function sim = cosineSim(vec1, vec2)
    denom = norm(vec1) * norm(vec2);
    if denom == 0
        sim = 0;
    else
        sim = dot(vec1, vec2) / denom;
    end
end

% --- Predict by Vector Similarity ---
function nextWord = predictByVector(coMatrix, vocab, word, topK)
    if nargin < 4, topK = 3; end
    wordIndex = containers.Map(vocab, 1:numel(vocab));
    if ~isKey(wordIndex, word)
        nextWord = {};
        fprintf('Word "%s" not found.\n', word);
        return;
    end
    widx = wordIndex(word);
    vec  = coMatrix(widx, :);
    sims = zeros(1, numel(vocab));
    for i = 1:numel(vocab)
        sims(i) = cosineSim(vec, coMatrix(i,:));
    end
    sims(widx)              = -inf;
    [sortedSims, sortedIdx] = sort(sims, 'descend');
    
    dotMask    = ~strcmp(vocab(sortedIdx), '.') & ~strcmp(vocab(sortedIdx), '។');
    sortedIdx  = sortedIdx(dotMask);
    sortedSims = sortedSims(dotMask);

    topK     = min(topK, numel(vocab));
    nextWord = vocab(sortedIdx(1:topK));
    fprintf('\nVector predictions after "%s":\n', word);
    for k = 1:topK
        fprintf('  %d. %-15s (sim=%.4f)\n', k, nextWord{k}, sortedSims(k));
    end
end

% ============================================================
% RUN PART 3
% ============================================================
fprintf('=== PART 3: Vector Representations ===\n\n');

% --- One-Hot Encoding ---
fprintf('--- One-Hot Encoding ---\n');
oneHot  = oneHotEncode(corpus.vocab);
exWords = corpus.vocab(1:min(3, numel(corpus.vocab)));
for i = 1:numel(exWords)
    vec = oneHot(exWords{i});
    fprintf('"%s" → [', exWords{i});
    fprintf('%d ', vec(1:min(10,end)));
    fprintf('...]\n');
end
fprintf('\n');

% --- Co-occurrence Matrix ---
fprintf('--- Co-occurrence Matrix ---\n');
coMatrix = buildCoOccurrence(trainTok, corpus.vocab, 2);
fprintf('Matrix size: %dx%d\n', size(coMatrix,1), size(coMatrix,2));
fprintf('\n');

% --- Co-occurrence for "cat" ---
fprintf('--- Co-occurrence counts for "cat" ---\n');
wordIndex = containers.Map(corpus.vocab, 1:numel(corpus.vocab));
sampleWord = corpus.vocab{2};
if isKey(wordIndex, sampleWord)
    catIdx = wordIndex(sampleWord);
    catRow = coMatrix(catIdx, :);
    [sortedC, sortedI] = sort(catRow, 'descend');  % ← add this line
    fprintf('Words most common near "%s":\n', sampleWord);
    for k = 1:min(5, numel(corpus.vocab))
        if sortedC(k) > 0
            fprintf('  %-15s count = %.0f\n', corpus.vocab{sortedI(k)}, sortedC(k));
        end
    end
end
fprintf('\n');

% --- Vector Similarity Predictions ---
fprintf('--- Vector Similarity Predictions ---\n');
testWords = corpus.vocab(2:min(4, numel(corpus.vocab)));
for i = 1:numel(testWords)
    predictByVector(coMatrix, corpus.vocab, testWords{i}, 3);
end
fprintf('\n');

% --- Compare Bigram vs Vector (inline - no external function) ---
fprintf('--- Comparison: Bigram vs Vector ---\n');
compareWords = corpus.vocab(2:min(4, numel(corpus.vocab)));
fprintf('%-10s %-20s %-20s\n', 'Word', 'Bigram predicts', 'Vector predicts');
fprintf('%s\n', repmat('-',1,50));

for i = 1:numel(compareWords)
    w = compareWords{i};

    % ---- bigram inline ----
    if ~isKey(bigramModel.bigram, w)
        bigramPred = 'unknown';
    else
        inner      = bigramModel.bigram(w);
        ws         = keys(inner);
        counts     = cell2mat(values(inner));

        dotMask = ~strcmp(ws, '.');
        ws      = ws(dotMask);
        counts  = counts(dotMask);

        [~, idx]   = max(counts);
        bigramPred = ws{idx};
    end

    % ---- vector inline ----
    if ~isKey(wordIndex, w)
        vectorPred = 'unknown';
    else
        widx = wordIndex(w);
        vec  = coMatrix(widx, :);
        sims = zeros(1, numel(corpus.vocab));
        for j = 1:numel(corpus.vocab)
            other = coMatrix(j,:);
            denom = norm(vec) * norm(other);
            if denom > 0
                sims(j) = dot(vec, other) / denom;
            end
        end
        sims(widx)   = -inf;
        [~, bestIdx] = max(sims);
        vectorPred   = corpus.vocab{bestIdx};
    end
    fprintf('%-10s %-20s %-20s\n', w, bigramPred, vectorPred);
end
fprintf('\n');

