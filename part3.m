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
function nextWord = predictByVector(coMatrix, vocab, words, topK)
    if nargin < 4, topK = 3; end
    % accept string or cell array
    if ischar(words), words = strsplit(words); end
    wordIndex = containers.Map(vocab, 1:numel(vocab));

    % average all input word vectors
    combined   = zeros(1, numel(vocab));
    validCount = 0;
    for wi = 1:numel(words)
        if isKey(wordIndex, words{wi})
            combined = combined + coMatrix(wordIndex(words{wi}), :);
            validCount = validCount + 1;
        else
            fprintf('Word "%s" not found.\n', words{wi});
        end
    end

    if validCount == 0
        nextWord = {};
        return;
    end
    combined = combined / validCount;

    % compute similarity
    sims = zeros(1, numel(vocab));
    for i = 1:numel(vocab)
        sims(i) = cosineSim(combined, coMatrix(i,:));
    end

    % exclude input words and dots
    for wi = 1:numel(words)
        if isKey(wordIndex, words{wi})
            sims(wordIndex(words{wi})) = -inf;
        end
    end
    dotMask    = ~strcmp(vocab, '.') & ~strcmp(vocab, '។');
    sims(~dotMask) = -inf;

    [sortedSims, sortedIdx] = sort(sims, 'descend');
    topK     = min(topK, numel(vocab));
    nextWord = vocab(sortedIdx(1:topK));

    fprintf('\nVector predictions after "%s":\n', strjoin(words, ' '));
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
    fprintf('"%s" -> [', exWords{i});
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
% pick first real word (not dot)
sampleWord = '';
for si = 1:numel(corpus.vocab)
    w = corpus.vocab{si};
    if ~strcmp(w, '.') && ~strcmp(w, '។') && numel(w) > 1
        sampleWord = w;
        break;
    end
end
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


% --- Vector Similarity Predictions ---
% single word examples
testWords = {'cat', 'the', 'bird'};
for i = 1:numel(testWords)
    predictByVector(coMatrix, corpus.vocab, testWords{i}, 3);
end
fprintf("\n");
% --- Compare Bigram vs Vector ---
fprintf('--- Comparison: Bigram vs Vector ---\n');
compareWords = {'cat', 'the', 'a'};
fprintf('%-10s %-20s %-20s\n', 'Word', 'Bigram predicts', 'Vector predicts');
fprintf('%s\n', repmat('-',1,50));
for i = 1:numel(compareWords)
    w = compareWords{i};

    % bigram
    if ~isKey(bigramModel.bigram, w)
        bigramPred = 'unknown';
    else
        inner   = bigramModel.bigram(w);
        ws      = keys(inner);
        counts  = cell2mat(values(inner));
        dotMask = ~strcmp(ws, '.') & ~strcmp(ws, '។');
        ws      = ws(dotMask);
        counts  = counts(dotMask);
        if isempty(ws)
            bigramPred = 'unknown';
        else
            [~, idx]   = max(counts);
            bigramPred = ws{idx};
        end
    end

    % vector
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
        sims(widx) = -inf;
        dotMask    = ~strcmp(corpus.vocab, '.') & ~strcmp(corpus.vocab, '។');
        sims(~dotMask) = -inf;
        [~, bestIdx] = max(sims);
        vectorPred   = corpus.vocab{bestIdx};
    end
    fprintf('%-14s %-30s %-10s\n', w, bigramPred, vectorPred);
end
fprintf('\n');
