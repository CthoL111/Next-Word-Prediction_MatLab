% ============================================================
% PART 4: Model Evaluation
% ============================================================

% --- Bigram Accuracy ---
function acc = evaluateBigram(model, testTokens)
    correct = 0;
    total   = 0;
    for i = 1:numel(testTokens)-1
        w1     = testTokens{i};
        actual = testTokens{i+1};
        if strcmp(actual, '.'), total = total + 1; continue; end
        if ~isKey(model.bigram, w1), total = total + 1; continue; end

        inner     = model.bigram(w1);
        ws        = keys(inner);
        counts    = cell2mat(values(inner));
        dotMask   = ~strcmp(ws, '.');
        ws        = ws(dotMask);
        counts    = counts(dotMask);
        if isempty(ws), total = total + 1; continue; end

        [~, idx]  = max(counts);
        predicted = ws{idx};
        if strcmp(predicted, actual), correct = correct + 1; end
        total = total + 1;
    end
    acc = correct / total * 100;
end

% --- Trigram Accuracy ---
function acc = evaluateTrigram(model, testTokens)
    correct = 0;
    total   = 0;
    for i = 1:numel(testTokens)-2
        w1     = testTokens{i};
        w2     = testTokens{i+1};
        actual = testTokens{i+2};
        if strcmp(actual, '.'), total = total + 1; continue; end
        key = [w1 ' ' w2];
        if ~isKey(model.trigram, key), total = total + 1; continue; end

        inner     = model.trigram(key);
        ws        = keys(inner);
        counts    = cell2mat(values(inner));
        dotMask   = ~strcmp(ws, '.');
        ws        = ws(dotMask);
        counts    = counts(dotMask);
        if isempty(ws), total = total + 1; continue; end

        [~, idx]  = max(counts);
        predicted = ws{idx};
        if strcmp(predicted, actual), correct = correct + 1; end
        total = total + 1;
    end
    acc = correct / total * 100;
end

% --- Vector Accuracy ---
function acc = evaluateVector(coMatrix, vocab, testTokens)
    wordIndex = containers.Map(vocab, 1:numel(vocab));
    correct   = 0;
    total     = 0;
    for i = 1:numel(testTokens)-1
        w1     = testTokens{i};
        actual = testTokens{i+1};
        if strcmp(actual, '.'), total = total + 1; continue; end
        if ~isKey(wordIndex, w1), total = total + 1; continue; end

        widx = wordIndex(w1);
        vec  = coMatrix(widx, :);
        sims = zeros(1, numel(vocab));
        for j = 1:numel(vocab)
            denom = norm(vec) * norm(coMatrix(j,:));
            if denom > 0
                sims(j) = dot(vec, coMatrix(j,:)) / denom;
            end
        end
        sims(widx) = -inf;
        [sortedSims, sortedIdx] = sort(sims, 'descend');
        dotMask    = ~strcmp(vocab(sortedIdx), '.');
        sortedIdx  = sortedIdx(dotMask);
        sortedSims = sortedSims(dotMask);
        predicted  = vocab{sortedIdx(1)};
        if strcmp(predicted, actual), correct = correct + 1; end
        total = total + 1;
    end
    acc = correct / total * 100;
end

% --- Perplexity (Bigram + Laplace) ---
function pp = measurePerplexity(model, testTokens, vocabSize)
    logProb = 0;
    N       = 0;
    for i = 1:numel(testTokens)-1
        w1 = testTokens{i};
        w2 = testTokens{i+1};
        if strcmp(w2, '.'), continue; end
        if isKey(model.bigram, w1)
            inner = model.bigram(w1);
            count = 0;
            if isKey(inner, w2), count = inner(w2); end
            total = model.unigram(w1);
        else
            count = 0; total = 0;
        end
        prob    = (count + 1) / (total + vocabSize);
        logProb = logProb + log2(prob);
        N       = N + 1;
    end
    pp = 2^(-logProb / N);
end

% ============================================================
% RUN PART 4
% ============================================================
fprintf('=== PART 4: Model Evaluation ===\n\n');

% --- 1. Train/Test Split ---
fprintf('--- Train/Test Split ---\n');
fprintf('Total : %d tokens\n', numel(corpus.tokens));
fprintf('Train : %d tokens (80%%)\n', numel(trainTok));
fprintf('Test  : %d tokens (20%%)\n\n', numel(testTok));

% --- 2. Accuracy ---
fprintf('--- Prediction Accuracy ---\n');
bigramAcc  = evaluateBigram(bigramModel, testTok);
trigramAcc = evaluateTrigram(trigramModel, testTok);
vectorAcc  = evaluateVector(coMatrix, corpus.vocab, testTok);
fprintf('Bigram  : %.2f%%\n', bigramAcc);
fprintf('Trigram : %.2f%%\n', trigramAcc);
fprintf('Vector  : %.2f%%\n\n', vectorAcc);

% --- 3. Perplexity ---
fprintf('--- Perplexity (lower = better) ---\n');
pp = measurePerplexity(bigramModel, testTok, corpus.vocabSize);
fprintf('Bigram Perplexity : %.4f\n\n', pp);

% --- 4. Comparison Table ---
fprintf('--- Comparison Table ---\n');
fprintf('%-10s %-10s %-30s\n', 'Model', 'Accuracy', 'Notes');
fprintf('%s\n', repmat('-', 1, 50));
fprintf('%-10s %-10s %-30s\n', 'Bigram',  sprintf('%.2f%%', bigramAcc),  'Uses 1 word context');
fprintf('%-10s %-10s %-30s\n', 'Trigram', sprintf('%.2f%%', trigramAcc), 'Uses 2 words context');
fprintf('%-10s %-10s %-30s\n', 'Vector',  sprintf('%.2f%%', vectorAcc),  'Uses word similarity');
fprintf('\n');

% --- 5. Best Model ---
fprintf('--- Best Model ---\n');
accs  = [bigramAcc, trigramAcc, vectorAcc];
names = {'Bigram', 'Trigram', 'Vector'};
[~, best] = max(accs);
fprintf('Winner : %s (%.2f%%)\n\n', names{best}, accs(best));
