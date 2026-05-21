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
if isKey(wordIndex, 'cat')
    catIdx = wordIndex('cat');
    catRow = coMatrix(catIdx, :);
    [sortedC, sortedI] = sort(catRow, 'descend');
    fprintf('Words most common near "cat":\n');
    for k = 1:min(5, numel(corpus.vocab))
        if sortedC(k) > 0
            fprintf('  %-15s count = %.0f\n', corpus.vocab{sortedI(k)}, sortedC(k));
        end
    end
end
fprintf('\n');

% --- Vector Similarity Predictions ---
fprintf('--- Vector Similarity Predictions ---\n');
testWords = {'cat', 'fish', 'bird'};
for i = 1:numel(testWords)
    predictByVector(coMatrix, corpus.vocab, testWords{i}, 3);
end
fprintf('\n');

% --- Compare Bigram vs Vector (inline - no external function) ---
fprintf('--- Comparison: Bigram vs Vector ---\n');
compareWords = {'cat', 'the', 'a'};
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
% ============================================================
% INTERACTIVE LOOP (updated: 1 word = bigram+vector, 2 words = trigram+vector)
% ============================================================
fprintf('=== INTERACTIVE WORD TO VECTOR ===\n');
fprintf('1 word input  → Bigram  + Vector prediction\n');
fprintf('2 word input  → Trigram + Vector prediction\n');
fprintf('Type "quit"   → Stop\n\n');

wordIdx = containers.Map(corpus.vocab, 1:corpus.vocabSize);

while true
    userInput = input('Enter word(s): ', 's');
    userInput = strtrim(lower(userInput));

    if strcmp(userInput, 'quit')
        fprintf('Goodbye!\n');
        break;
    end

    inputWords = strsplit(userInput);
    numWords   = numel(inputWords);

    % check all words are in vocabulary
    allFound = true;
    for w = 1:numWords
        if ~isKey(wordIdx, inputWords{w})
            fprintf('[!] "%s" not in vocabulary. Try: %s\n\n', ...
                    inputWords{w}, strjoin(corpus.vocab(1:min(5,end)), ', '));
            allFound = false;
        end
    end
    if ~allFound, continue; end

    % ============================================================
    if numWords == 1
        userWord = inputWords{1};
        idx      = wordIdx(userWord);

        % --- One-Hot Vector ---
        fprintf('\n--- One-Hot Vector for "%s" ---\n', userWord);
        vec = oneHot(userWord);
        fprintf('[');
        fprintf('%d ', vec(1:min(10, end)));
        if corpus.vocabSize > 10
            fprintf('... (%d dims total)', corpus.vocabSize);
        end
        fprintf(']\nPosition %d = 1, all others = 0\n\n', idx);

        % --- Co-occurrence ---
        fprintf('--- Co-occurrence Vector for "%s" ---\n', userWord);
        row = coMatrix(idx, :);
        [sorted, sortedI] = sort(row, 'descend');
        for k = 1:min(5, corpus.vocabSize)
            if sorted(k) <= 0, break; end
            fprintf('  %-15s count = %.0f\n', corpus.vocab{sortedI(k)}, sorted(k));
        end

        % --- Top 5 Similar ---
        fprintf('\n--- Top 5 Similar Words to "%s" ---\n', userWord);
        vec  = coMatrix(idx, :);
        sims = zeros(1, corpus.vocabSize);
        for i = 1:corpus.vocabSize
            sims(i) = cosineSim(vec, coMatrix(i,:));
        end
        sims(idx) = -inf;
        [sortedSims, sortedIdx] = sort(sims, 'descend');
        for k = 1:min(5, corpus.vocabSize-1)
            fprintf('  %d. %-15s sim = %.4f\n', k, corpus.vocab{sortedIdx(k)}, sortedSims(k));
        end

        % --- Prediction ---
        fprintf('\n--- Next Word Prediction for "%s" ---\n', userWord);

        % Bigram
        if isKey(bigramModel.bigram, userWord)
            inner   = bigramModel.bigram(userWord);
            ws      = keys(inner);
            wCounts = cell2mat(values(inner));
            [~, bi] = max(wCounts);
            biPred  = ws{bi};
            total   = bigramModel.unigram(userWord);
            biProb  = inner(biPred) / total;
            fprintf('Bigram  : "%s" → "%s" (prob: %.4f)\n', userWord, biPred, biProb);
        else
            fprintf('Bigram  : "%s" → unknown\n', userWord);
        end

        % Vector (single word row)
        vecPred = corpus.vocab{sortedIdx(1)};
        fprintf('Vector  : "%s" → "%s" (sim:  %.4f)\n\n', userWord, vecPred, sortedSims(1));

    % ============================================================
    elseif numWords >= 2
        word1 = inputWords{numWords-1};
        word2 = inputWords{numWords};
        idx1  = wordIdx(word1);
        idx2  = wordIdx(word2);

        % --- Combined Vector (average of both word vectors) ---
        fprintf('\n--- Combined Vector for "%s %s" ---\n', word1, word2);
        vec1     = coMatrix(idx1, :);
        vec2     = coMatrix(idx2, :);
        combined = (vec1 + vec2) / 2;  % average = context vector
        fprintf('(Average of "%s" and "%s" co-occurrence vectors)\n\n', word1, word2);

        % --- Top 5 Similar to combined vector ---
        fprintf('--- Top 5 Similar Words to "%s %s" ---\n', word1, word2);
        sims = zeros(1, corpus.vocabSize);
        for i = 1:corpus.vocabSize
            sims(i) = cosineSim(combined, coMatrix(i,:));
        end
        sims(idx1) = -inf;  % exclude input words
        sims(idx2) = -inf;
        [sortedSims, sortedIdx] = sort(sims, 'descend');
        for k = 1:min(5, corpus.vocabSize-1)
            fprintf('  %d. %-15s sim = %.4f\n', k, corpus.vocab{sortedIdx(k)}, sortedSims(k));
        end

        % --- Prediction ---
        fprintf('\n--- Next Word Prediction for "%s %s" ---\n', word1, word2);

        % Trigram
        triKey = [word1 ' ' word2];
        if isKey(trigramModel.trigram, triKey)
            inner   = trigramModel.trigram(triKey);
            ws      = keys(inner);
            wCounts = cell2mat(values(inner));
            [~, ti] = max(wCounts);
            triPred = ws{ti};
            fprintf('Trigram : "%s %s" → "%s"\n', word1, word2, triPred);
        else
            fprintf('Trigram : "%s %s" → unknown (pair not seen in training)\n', word1, word2);
        end

        % Vector (combined/averaged)
        vecPred = corpus.vocab{sortedIdx(1)};
        fprintf('Vector  : "%s %s" → "%s" (sim:  %.4f)\n\n', word1, word2, vecPred, sortedSims(1));
    end
end