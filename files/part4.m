% ============================================================
% PART 4: Model Evaluation (15%)
% ============================================================
 
% --- 1. Calculate Prediction Accuracy ---
function acc = evaluateAccuracy(model, testTokens, modelType)
    correct = 0;
    total   = 0;

    if strcmp(modelType, 'bigram')
        for i = 1:numel(testTokens)-1
            w1     = testTokens{i};
            actual = testTokens{i+1};

            % predict next word
            if ~isKey(model.bigram, w1)
                total = total + 1;
                continue;
            end
            inner      = model.bigram(w1);
            nextWords  = keys(inner);
            nextCounts = cell2mat(values(inner));
            [~, idx]   = max(nextCounts);
            predicted  = nextWords{idx};

            if strcmp(predicted, actual)
                correct = correct + 1;
            end
            total = total + 1;
        end

    elseif strcmp(modelType, 'trigram')
        for i = 1:numel(testTokens)-2
            w1     = testTokens{i};
            w2     = testTokens{i+1};
            actual = testTokens{i+2};
            key    = [w1 ' ' w2];

            if ~isKey(model.trigram, key)
                total = total + 1;
                continue;
            end
            inner      = model.trigram(key);
            nextWords  = keys(inner);
            nextCounts = cell2mat(values(inner));
            [~, idx]   = max(nextCounts);
            predicted  = nextWords{idx};

            if strcmp(predicted, actual)
                correct = correct + 1;
            end
            total = total + 1;
        end
    end

    acc = correct / total * 100;
    fprintf('%-10s Accuracy: %.2f%%  (%d/%d correct)\n', modelType, acc, correct, total);
end

% --- 2. Measure Perplexity ---
function pp = measurePerplexity(model, testTokens, vocabSize)
    logProb = 0;
    N       = numel(testTokens) - 1;

    for i = 1:N
        w1 = testTokens{i};
        w2 = testTokens{i+1};

        % use Laplace smoothing so never log(0)
        if isKey(model.bigram, w1)
            inner = model.bigram(w1);
            if isKey(inner, w2)
                count = inner(w2);
            else
                count = 0;
            end
            total = model.unigram(w1);
        else
            count = 0;
            total = 0;
        end

        prob    = (count + 1) / (total + vocabSize);
        logProb = logProb + log2(prob);
    end

    pp = 2^(-logProb / N);
    fprintf('%-10s Perplexity: %.4f\n', 'bigram', pp);
end

% --- 3. Vector Accuracy ---
function acc = evaluateVectorAccuracy(coMatrix, vocab, testTokens)
    wordIndex = containers.Map(vocab, 1:numel(vocab));
    correct   = 0;
    total     = 0;

    for i = 1:numel(testTokens)-1
        w1     = testTokens{i};
        actual = testTokens{i+1};

        if ~isKey(wordIndex, w1), total = total+1; continue; end

        widx = wordIndex(w1);
        vec  = coMatrix(widx, :);

        % find most similar word
        sims = zeros(1, numel(vocab));
        for j = 1:numel(vocab)
            other = coMatrix(j,:);
            denom = norm(vec) * norm(other);
            if denom > 0
                sims(j) = dot(vec,other) / denom;
            end
        end
        sims(widx)   = -inf;
        [~, bestIdx] = max(sims);
        predicted    = vocab{bestIdx};

        if strcmp(predicted, actual)
            correct = correct + 1;
        end
        total = total + 1;
    end

    acc = correct / total * 100;
    fprintf('%-10s Accuracy: %.2f%%  (%d/%d correct)\n', 'vector', acc, correct, total);
end

% ============================================================
% RUN PART 4
% ============================================================
fprintf('=== PART 4: Model Evaluation ===\n\n');

% --- Step 1: Split already done in Part 2 ---
fprintf('--- 1. Train/Test Split ---\n');
fprintf('Train tokens : %d\n', numel(trainTok));
fprintf('Test tokens  : %d\n', numel(testTok));
fprintf('\n');

% --- Step 2: Accuracy ---
fprintf('--- 2. Prediction Accuracy ---\n');
bigramAcc  = evaluateAccuracy(bigramModel,  testTok, 'bigram');
trigramAcc = evaluateAccuracy(trigramModel, testTok, 'trigram');
vectorAcc  = evaluateVectorAccuracy(coMatrix, corpus.vocab, testTok);
fprintf('\n');

% --- Step 3: Perplexity ---
fprintf('--- 3. Perplexity (lower = better) ---\n');
measurePerplexity(bigramModel, testTok, corpus.vocabSize);
fprintf('\n');

% --- Step 4: Compare All Approaches ---
fprintf('--- 4. Comparison of All Approaches ---\n');
fprintf('%-15s %-15s %-30s\n', 'Approach', 'Accuracy', 'Notes');
fprintf('%s\n', repmat('-',1,60));
fprintf('%-15s %-15s %-30s\n', 'Bigram', ...
    sprintf('%.2f%%', bigramAcc), ...
    'Fast, exact pairs only');
fprintf('%-15s %-15s %-30s\n', 'Trigram', ...
    sprintf('%.2f%%', trigramAcc), ...
    'More context, needs more data');
fprintf('%-15s %-15s %-30s\n', 'Vector', ...
    sprintf('%.2f%%', vectorAcc), ...
    'Finds similar words by meaning');
fprintf('\n');

% --- Step 5: Strengths and Weaknesses ---
fprintf('--- 5. Strengths and Weaknesses ---\n\n');

fprintf('BIGRAM:\n');
fprintf('  Strengths : simple, fast, easy to build\n');
fprintf('  Weaknesses: only looks at 1 word, misses context\n\n');

fprintf('TRIGRAM:\n');
fprintf('  Strengths : uses 2 words = more context = better prediction\n');
fprintf('  Weaknesses: needs more training data to work well\n\n');

fprintf('VECTOR:\n');
fprintf('  Strengths : understands word similarity and meaning\n');
fprintf('  Weaknesses: slower, needs large corpus to be accurate\n\n');